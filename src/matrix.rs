//! Dense (unquantized) matrix for `.bin` models.

use std::io::Read;

use crate::error::{Error, Result};
use crate::io::BinaryReader;

/// A dense row-major matrix of `f32` values.
///
/// Used for both input and output weight matrices in unquantized `.bin`
/// models, and for the output matrix in most `.ftz` models (which
/// typically only quantize the input matrix).
#[derive(Debug)]
pub(crate) struct DenseMatrix {
   pub m: usize,
   pub n: usize,
   data: Vec<f32>,
}

impl DenseMatrix {
   /// Load a dense matrix from the binary format.
   ///
   /// Reads two `i64` dimensions (rows, cols) followed by `rows * cols`
   /// little-endian `f32` values.
   pub fn load<R: Read>(reader: &mut BinaryReader<R>) -> Result<Self> {
      let m = reader.read_i64_as_usize()?;
      let n = reader.read_i64_as_usize()?;
      let len = m
         .checked_mul(n)
         .ok_or_else(|| Error::InvalidModel("matrix dimensions overflow".to_string()))?;
      let data = reader.read_f32_vec(len)?;
      Ok(Self { m, n, data })
   }

   /// Add row `i` of this matrix to the vector `x` element-wise.
   pub fn add_row_to_vec(&self, x: &mut [f32], row: usize) {
      let start = row * self.n;
      for (xi, &ri) in x.iter_mut().zip(self.data[start..].iter()) {
         *xi += ri;
      }
   }

   /// Compute the dot product of row `i` with the vector `vec`.
   pub fn dot_row(&self, vec: &[f32], row: usize) -> f32 {
      let start = row * self.n;
      self.data[start..start + self.n]
         .iter()
         .zip(vec.iter())
         .map(|(&a, &b)| a * b)
         .sum()
   }
}
