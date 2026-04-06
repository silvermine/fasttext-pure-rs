//! # fasttext-pure-rs
//!
//! A pure-Rust fastText inference engine for language identification and text
//! classification. Loads `.bin` and `.ftz` (quantized) model files produced by
//! Facebook's [fastText](https://fasttext.cc/) and runs prediction without any
//! C/C++ dependencies.
//!
//! ## Quick Start
//!
//! ```no_run
//! use fasttext_pure_rs::FastText;
//!
//! let bytes = std::fs::read("lid.176.ftz").unwrap();
//! let model = FastText::load_from_reader(std::io::Cursor::new(bytes)).unwrap();
//! let predictions = model.predict("hello world", 1, 0.0).unwrap();
//!
//! for prediction in &predictions {
//!    println!("{}: {:.4}", prediction.label, prediction.probability);
//! }
//! ```
//!
//! ## No-I/O Design
//!
//! The primary loading method is
//! [`load_from_reader`](FastText::load_from_reader), which accepts any
//! [`Read`](std::io::Read) implementor. The library itself never opens
//! files or makes network calls — callers control all I/O and pass
//! bytes in. This design follows the principle that libraries should be
//! I/O-free so that the same code works in server, Lambda, desktop,
//! mobile, and WASM environments without modification.
//!
//! A convenience [`load`](FastText::load) method that opens a file by
//! path is available behind the **`std`** feature (enabled by default).
//! Disable default features to compile without any filesystem
//! dependency:
//!
//! ```toml
//! [dependencies]
//! fasttext-pure-rs = { version = "0.1", default-features = false }
//! ```
//!
//! ## Model Formats
//!
//! - **`.bin`** — unquantized models with dense weight matrices
//! - **`.ftz`** — quantized models using product quantization for
//!   compressed weight matrices (typically ~10x smaller)
//!
//! Both formats use binary format version 12, the standard for all modern
//! fastText models.

mod args;
mod dictionary;
mod error;
mod io;
mod matrix;
mod model;
mod quantized_matrix;

pub use error::{Error, Result};

#[cfg(feature = "std")]
use std::fs::File;
use std::io::Read;
#[cfg(feature = "std")]
use std::path::Path;

use args::ModelArgs;
use dictionary::Dictionary;
use io::BinaryReader;
use matrix::DenseMatrix;
use model::{Matrix, Model};
use quantized_matrix::QuantMatrix;

/// A single prediction result containing a label and its probability.
///
/// # Examples
///
/// ```no_run
/// use fasttext_pure_rs::FastText;
///
/// let bytes = std::fs::read("lid.176.ftz").unwrap();
/// let model = FastText::load_from_reader(
///    std::io::Cursor::new(bytes),
/// ).unwrap();
/// let predictions = model.predict("bonjour le monde", 1, 0.0).unwrap();
///
/// let best = &predictions[0];
/// assert!(best.label.starts_with("__label__"));
/// assert!(best.probability > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct Prediction {
   /// The predicted label (e.g., `"__label__en"`).
   pub label: String,

   /// The probability of the prediction, between 0.0 and 1.0.
   pub probability: f32,
}

/// A loaded fastText model for inference.
///
/// `FastText` is the main entry point for this crate. Load a model from
/// a [`Read`](std::io::Read) source (or a file path with the `std`
/// feature), then call [`predict`](Self::predict) or
/// [`get_sentence_vector`](Self::get_sentence_vector).
///
/// This struct is `Send + Sync`, making it safe to share across threads
/// via `Arc` in server contexts.
///
/// # Examples
///
/// ```no_run
/// use fasttext_pure_rs::FastText;
///
/// let bytes = std::fs::read("lid.176.ftz").unwrap();
/// let model = FastText::load_from_reader(
///    std::io::Cursor::new(bytes),
/// ).unwrap();
///
/// let predictions = model.predict("hello world", 3, 0.0).unwrap();
/// for p in &predictions {
///    println!("{}: {:.4}", p.label, p.probability);
/// }
/// ```
#[derive(Debug)]
pub struct FastText {
   args: ModelArgs,
   dictionary: Dictionary,
   model: Model,
}

// Compile-time assertion that FastText is Send + Sync
const _: fn() = || {
   fn assert_send_sync<T: Send + Sync>() {}
   assert_send_sync::<FastText>();
};

impl FastText {
   /// Load a fastText model from a file path.
   ///
   /// This is a convenience wrapper around
   /// [`load_from_reader`](Self::load_from_reader) that opens the file
   /// for you. Only available with the `std` feature (enabled by
   /// default).
   ///
   /// # Examples
   ///
   /// ```no_run
   /// use fasttext_pure_rs::FastText;
   ///
   /// let model = FastText::load("lid.176.ftz").unwrap();
   /// ```
   #[cfg(feature = "std")]
   pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
      let file = File::open(path)?;
      Self::load_from_reader(std::io::BufReader::new(file))
   }

   /// Load a fastText model from any [`Read`] implementor.
   ///
   /// Useful for loading models from embedded bytes (e.g.,
   /// `include_bytes!()` in Tauri apps) or from a network stream.
   ///
   /// # Examples
   ///
   /// ```no_run
   /// use fasttext_pure_rs::FastText;
   ///
   /// let bytes = std::fs::read("lid.176.ftz").unwrap();
   /// let model = FastText::load_from_reader(std::io::Cursor::new(bytes)).unwrap();
   /// ```
   pub fn load_from_reader<R: Read>(reader: R) -> Result<Self> {
      let mut reader = BinaryReader::new(reader);
      reader.validate_header()?;

      let args = ModelArgs::load(&mut reader)?;
      let dictionary = Dictionary::load(&mut reader, &args)?;

      // Input matrix: quantized for .ftz, dense for .bin
      let quant = reader.read_u8()? != 0;
      let input = if quant {
         Matrix::Quantized(QuantMatrix::load(&mut reader)?)
      } else {
         Matrix::Dense(DenseMatrix::load(&mut reader)?)
      };

      // Output matrix: usually dense, rarely quantized
      let qout = reader.read_u8()? != 0;
      let output = if quant && qout {
         Matrix::Quantized(QuantMatrix::load(&mut reader)?)
      } else {
         Matrix::Dense(DenseMatrix::load(&mut reader)?)
      };

      let label_counts = dictionary.get_label_counts();
      let model = Model::new(input, output, &args, &label_counts);

      Ok(Self {
         args,
         dictionary,
         model,
      })
   }

   /// Run prediction on the given text, returning the top `k` results.
   ///
   /// Only predictions with softmax probability >= `threshold` are
   /// returned. Pass `threshold = 0.0` to get all top-k predictions
   /// regardless of score.
   ///
   /// Returns [`Error::EmptyInput`] if the text produces no usable
   /// tokens.
   ///
   /// # Examples
   ///
   /// ```no_run
   /// use fasttext_pure_rs::FastText;
   ///
   /// let bytes = std::fs::read("lid.176.ftz").unwrap();
   /// let model = FastText::load_from_reader(
   ///    std::io::Cursor::new(bytes),
   /// ).unwrap();
   /// let predictions = model.predict("hello world", 1, 0.0).unwrap();
   /// assert!(!predictions.is_empty());
   /// ```
   pub fn predict(&self, text: &str, k: usize, threshold: f32) -> Result<Vec<Prediction>> {
      let (words, _labels) = self.dictionary.get_line(text);

      if words.is_empty() {
         return Err(Error::EmptyInput);
      }

      let results = self.model.predict(&words, k, threshold)?;

      Ok(results
         .into_iter()
         .map(|(log_prob, label_id)| Prediction {
            label: self.dictionary.get_label(label_id).to_string(),
            probability: log_prob.exp(),
         })
         .collect())
   }

   /// Compute the sentence embedding vector for the given text.
   ///
   /// Returns a vector of `dim` dimensions representing the average of
   /// all input word/subword embeddings for the tokens in the text.
   ///
   /// Returns [`Error::EmptyInput`] if the text produces no usable
   /// tokens.
   ///
   /// # Examples
   ///
   /// ```no_run
   /// use fasttext_pure_rs::FastText;
   ///
   /// let bytes = std::fs::read("lid.176.ftz").unwrap();
   /// let model = FastText::load_from_reader(
   ///    std::io::Cursor::new(bytes),
   /// ).unwrap();
   /// let vector = model.get_sentence_vector("hello world").unwrap();
   /// ```
   pub fn get_sentence_vector(&self, text: &str) -> Result<Vec<f32>> {
      let (words, _labels) = self.dictionary.get_line(text);

      if words.is_empty() {
         return Err(Error::EmptyInput);
      }

      self.model.compute_hidden_vec(&words)
   }

   /// Return the dimensionality of the model's embedding vectors.
   ///
   /// This is the length of vectors returned by
   /// [`get_sentence_vector`](Self::get_sentence_vector). For
   /// `lid.176.ftz`, the dimension is 16.
   ///
   /// # Examples
   ///
   /// ```no_run
   /// use fasttext_pure_rs::FastText;
   ///
   /// let bytes = std::fs::read("lid.176.ftz").unwrap();
   /// let model = FastText::load_from_reader(
   ///    std::io::Cursor::new(bytes),
   /// ).unwrap();
   /// assert_eq!(model.dim(), 16);
   /// ```
   pub fn dim(&self) -> usize {
      self.args.dim as usize
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn fast_text_is_send_sync() {
      fn assert_send_sync<T: Send + Sync>() {}
      assert_send_sync::<FastText>();
   }
}
