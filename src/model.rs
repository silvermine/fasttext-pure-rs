//! Inference engine for fastText prediction and sentence embedding.

use crate::args::{LossType, ModelArgs};
use crate::error::Result;
use crate::matrix::DenseMatrix;
use crate::quantized_matrix::QuantMatrix;

/// A weight matrix that can be either dense (`.bin`) or quantized
/// (`.ftz`).
#[derive(Debug)]
pub(crate) enum Matrix {
   Dense(DenseMatrix),
   Quantized(QuantMatrix),
}

impl Matrix {
   /// Add row `row` of this matrix to the vector `x`.
   fn add_row_to_vec(&self, x: &mut [f32], row: usize) -> Result<()> {
      match self {
         Self::Dense(m) => {
            m.add_row_to_vec(x, row);
            Ok(())
         }
         Self::Quantized(m) => m.add_to_vector(x, row),
      }
   }

   /// Compute the dot product of `vec` with row `row`.
   fn dot_row(&self, vec: &[f32], row: usize) -> Result<f32> {
      match self {
         Self::Dense(m) => Ok(m.dot_row(vec, row)),
         Self::Quantized(m) => m.dot_row(vec, row),
      }
   }

   /// Number of rows in this matrix.
   pub fn rows(&self) -> usize {
      match self {
         Self::Dense(m) => m.m,
         Self::Quantized(m) => m.m,
      }
   }
}

/// A node in the Huffman tree for hierarchical softmax.
///
/// Only `left`, `right`, and `count` are needed for inference. The C++
/// implementation also stores `parent` and `binary`, but those are only
/// used during training.
#[derive(Debug, Clone)]
struct TreeNode {
   left: i32,
   right: i32,
   count: i64,
}

/// The fastText inference model holding input/output matrices and
/// performing prediction via softmax or hierarchical softmax.
#[derive(Debug)]
pub(crate) struct Model {
   input: Matrix,
   output: Matrix,
   osz: usize,
   hsz: usize,
   loss: LossType,
   tree: Vec<TreeNode>,
}

impl Model {
   /// Create a new `Model` from loaded matrices, args, and label counts.
   ///
   /// For hierarchical softmax models, the Huffman tree is built from
   /// the label counts at construction time.
   pub fn new(input: Matrix, output: Matrix, args: &ModelArgs, label_counts: &[i64]) -> Self {
      let osz = output.rows();
      let mut model = Self {
         input,
         output,
         osz,
         hsz: args.dim as usize,
         loss: args.loss,
         tree: Vec::new(),
      };

      if args.loss == LossType::HierarchicalSoftmax {
         model.build_tree(label_counts);
      }

      model
   }

   /// Build the Huffman tree from label counts.
   ///
   /// Replicates the C++ `Model::buildTree` algorithm exactly: leaves
   /// are labels (indices 0..osz), internal nodes are indices osz..2*osz-2,
   /// and the root is at index 2*osz-2.
   fn build_tree(&mut self, counts: &[i64]) {
      let tree_size = 2 * self.osz - 1;
      self.tree = vec![
         TreeNode {
            left: -1,
            right: -1,
            count: i64::MAX,
         };
         tree_size
      ];

      // Initialize leaf nodes with label counts
      for (i, &count) in counts.iter().enumerate().take(self.osz) {
         self.tree[i].count = count;
      }

      // Build tree bottom-up by greedily merging the two smallest nodes
      let mut leaf = self.osz as i32 - 1;
      let mut node = self.osz;

      for i in self.osz..tree_size {
         let mut mini = [0i32; 2];
         for slot in &mut mini {
            if leaf >= 0 && self.tree[leaf as usize].count < self.tree[node].count {
               *slot = leaf;
               leaf -= 1;
            } else {
               *slot = node as i32;
               node += 1;
            }
         }
         self.tree[i].left = mini[0];
         self.tree[i].right = mini[1];
         self.tree[i].count = self.tree[mini[0] as usize].count + self.tree[mini[1] as usize].count;
      }
   }

   /// Run top-k prediction on tokenized input.
   ///
   /// Returns a vector of `(log_probability, label_index)` pairs sorted
   /// by descending probability.
   pub fn predict(&self, input: &[i32], k: usize, threshold: f32) -> Result<Vec<(f32, i32)>> {
      let mut hidden = vec![0.0f32; self.hsz];
      self.fill_hidden(input, &mut hidden)?;

      let mut heap: Vec<(f32, i32)> = Vec::with_capacity(k + 1);

      if self.loss == LossType::HierarchicalSoftmax {
         let root = (2 * self.osz - 2) as i32;
         self.dfs(k, threshold, root, 0.0, &mut heap, &hidden)?;
      } else {
         self.find_k_best(k, threshold, &mut heap, &hidden)?;
      }

      // Sort by descending log probability
      heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
      Ok(heap)
   }

   /// Depth-first search through the Huffman tree for hierarchical
   /// softmax prediction.
   ///
   /// The maximum recursion depth is `2 * osz - 2` (the height of the
   /// Huffman tree). For `lid.176.ftz` (176 labels), this is at most
   /// 350 frames deep. Models with >10,000 labels may require
   /// increasing the thread stack size on platforms with small defaults
   /// (e.g., some Android worker threads default to 64 KB).
   fn dfs(
      &self,
      k: usize,
      threshold: f32,
      node: i32,
      score: f32,
      heap: &mut Vec<(f32, i32)>,
      hidden: &[f32],
   ) -> Result<()> {
      // Prune if score can't beat current k-th best
      if heap.len() == k && score < heap[0].0 {
         return Ok(());
      }

      // Prune if score is below the log of threshold
      if threshold > 0.0 && score < std_log(threshold) {
         return Ok(());
      }

      let node_usize = node as usize;

      // Leaf node: record the prediction
      if self.tree[node_usize].left == -1 && self.tree[node_usize].right == -1 {
         heap.push((score, node));
         heap_push(heap);
         if heap.len() > k {
            heap_pop(heap);
         }
         return Ok(());
      }

      // Internal node: compute sigmoid and recurse
      let f = sigmoid(self.output.dot_row(hidden, node_usize - self.osz)?);

      self.dfs(
         k,
         threshold,
         self.tree[node_usize].left,
         score + std_log(1.0 - f),
         heap,
         hidden,
      )?;
      self.dfs(
         k,
         threshold,
         self.tree[node_usize].right,
         score + std_log(f),
         heap,
         hidden,
      )?;
      Ok(())
   }

   /// Find top-k predictions using softmax over all output labels.
   fn find_k_best(
      &self,
      k: usize,
      threshold: f32,
      heap: &mut Vec<(f32, i32)>,
      hidden: &[f32],
   ) -> Result<()> {
      let mut output = vec![0.0f32; self.osz];
      self.compute_output_softmax(hidden, &mut output)?;

      for (i, &prob) in output[..self.osz].iter().enumerate() {
         if prob < threshold {
            continue;
         }
         let log_prob = std_log(prob);

         if heap.len() == k && log_prob < heap[0].0 {
            continue;
         }

         heap.push((log_prob, i as i32));
         heap_push(heap);
         if heap.len() > k {
            heap_pop(heap);
         }
      }
      Ok(())
   }

   /// Compute the hidden (sentence embedding) vector for tokenized input.
   ///
   /// This is the average of all input embeddings for the tokens in the
   /// input.
   pub fn compute_hidden_vec(&self, input: &[i32]) -> Result<Vec<f32>> {
      let mut hidden = vec![0.0f32; self.hsz];
      self.fill_hidden(input, &mut hidden)?;
      Ok(hidden)
   }

   /// Fill the hidden vector with the average of input embeddings.
   fn fill_hidden(&self, input: &[i32], hidden: &mut [f32]) -> Result<()> {
      hidden.iter_mut().for_each(|x| *x = 0.0);
      for &idx in input {
         self.input.add_row_to_vec(hidden, idx as usize)?;
      }
      if !input.is_empty() {
         let inv = 1.0 / input.len() as f32;
         hidden.iter_mut().for_each(|x| *x *= inv);
      }
      Ok(())
   }

   /// Compute softmax over the output matrix, writing probabilities to
   /// `output`.
   fn compute_output_softmax(&self, hidden: &[f32], output: &mut [f32]) -> Result<()> {
      // Compute dot products of each output row with hidden
      for (i, out) in output.iter_mut().enumerate().take(self.osz) {
         *out = self.output.dot_row(hidden, i)?;
      }

      // Numerically stable softmax: subtract max, then exp and normalize
      let max = output[..self.osz]
         .iter()
         .copied()
         .fold(f32::NEG_INFINITY, f32::max);

      let mut z: f32 = 0.0;
      for val in output[..self.osz].iter_mut() {
         *val = (*val - max).exp();
         z += *val;
      }
      for val in output[..self.osz].iter_mut() {
         *val /= z;
      }
      Ok(())
   }
}

/// Compute `ln(x + 1e-5)`, matching the C++ `Model::std_log`.
fn std_log(x: f32) -> f32 {
   (x + 1e-5).ln()
}

/// Sigmoid activation function.
fn sigmoid(x: f32) -> f32 {
   if x < -8.0 {
      return 0.0;
   }
   if x > 8.0 {
      return 1.0;
   }
   1.0 / (1.0 + (-x).exp())
}

// Min-heap operations for top-k selection.
// The smallest element is at index 0, so we keep the k largest by
// evicting the minimum when the heap exceeds k.

/// Push the last element up to maintain min-heap order.
fn heap_push(heap: &mut [(f32, i32)]) {
   let mut idx = heap.len() - 1;
   while idx > 0 {
      let parent = (idx - 1) / 2;
      if heap[idx].0 < heap[parent].0 {
         heap.swap(idx, parent);
         idx = parent;
      } else {
         break;
      }
   }
}

/// Remove the minimum element (root) from a min-heap.
fn heap_pop(heap: &mut Vec<(f32, i32)>) {
   let last = heap.len() - 1;
   heap.swap(0, last);
   heap.pop();

   // Sift root down
   let len = heap.len();
   let mut idx = 0;
   loop {
      let left = 2 * idx + 1;
      let right = 2 * idx + 2;
      let mut smallest = idx;

      if left < len && heap[left].0 < heap[smallest].0 {
         smallest = left;
      }
      if right < len && heap[right].0 < heap[smallest].0 {
         smallest = right;
      }
      if smallest == idx {
         break;
      }
      heap.swap(idx, smallest);
      idx = smallest;
   }
}
