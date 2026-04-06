//! Error types for the fasttext-pure-rs crate.

use std::io;
use thiserror::Error;

/// Errors that can occur when loading or using a fastText model.
///
/// # Examples
///
/// ```no_run
/// use fasttext_pure_rs::FastText;
///
/// let bytes = std::fs::read("model.ftz").unwrap();
/// match FastText::load_from_reader(std::io::Cursor::new(bytes)) {
///    Ok(_) => println!("loaded"),
///    Err(e) => eprintln!("error: {e}"),
/// }
/// ```
#[derive(Debug, Error)]
pub enum Error {
   /// An I/O error occurred while reading the model file.
   #[error("I/O error: {0}")]
   Io(#[from] io::Error),

   /// The model file has an invalid format or is corrupted.
   #[error("invalid model format: {0}")]
   InvalidModel(String),

   /// The model file uses an unsupported binary format version.
   #[error("unsupported model version: expected {expected}, got {actual}")]
   UnsupportedVersion {
      /// The expected version number.
      expected: i32,
      /// The actual version number found in the file.
      actual: i32,
   },

   /// An empty string was passed to a function that requires non-empty
   /// input.
   #[error("empty input: no usable tokens found in input text")]
   EmptyInput,
}

/// A specialized `Result` type for fastText operations.
pub type Result<T> = std::result::Result<T, Error>;
