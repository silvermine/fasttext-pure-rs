//! Model arguments parsed from the binary format header.

use std::io::Read;

use crate::error::{Error, Result};
use crate::io::BinaryReader;

/// The type of model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelType {
   Cbow = 1,
   SkipGram = 2,
   Supervised = 3,
}

impl TryFrom<i32> for ModelType {
   type Error = Error;

   fn try_from(value: i32) -> Result<Self> {
      match value {
         1 => Ok(Self::Cbow),
         2 => Ok(Self::SkipGram),
         3 => Ok(Self::Supervised),
         _ => Err(Error::InvalidModel(format!("unknown model type: {value}"))),
      }
   }
}

/// The loss function used during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LossType {
   HierarchicalSoftmax = 1,
   NegativeSampling = 2,
   Softmax = 3,
}

impl TryFrom<i32> for LossType {
   type Error = Error;

   fn try_from(value: i32) -> Result<Self> {
      match value {
         1 => Ok(Self::HierarchicalSoftmax),
         2 => Ok(Self::NegativeSampling),
         3 => Ok(Self::Softmax),
         _ => Err(Error::InvalidModel(format!("unknown loss type: {value}"))),
      }
   }
}

/// Hyperparameters and configuration stored in the model file header.
///
/// These are the 13 fields serialized immediately after the magic number
/// and version: 12 `i32` values followed by 1 `f64`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ModelArgs {
   pub dim: i32,
   pub ws: i32,
   pub epoch: i32,
   pub min_count: i32,
   pub neg: i32,
   pub word_ngrams: i32,
   pub loss: LossType,
   pub model_type: ModelType,
   pub bucket: i32,
   pub minn: i32,
   pub maxn: i32,
   pub lr_update_rate: i32,
   pub t: f64,
}

impl ModelArgs {
   /// Deserialize `ModelArgs` from the binary format.
   pub fn load<R: Read>(reader: &mut BinaryReader<R>) -> Result<Self> {
      let dim = reader.read_i32()?;
      let ws = reader.read_i32()?;
      let epoch = reader.read_i32()?;
      let min_count = reader.read_i32()?;
      let neg = reader.read_i32()?;
      let word_ngrams = reader.read_i32()?;
      let loss = LossType::try_from(reader.read_i32()?)?;
      let model_type = ModelType::try_from(reader.read_i32()?)?;
      let bucket = reader.read_i32()?;
      let minn = reader.read_i32()?;
      let maxn = reader.read_i32()?;
      let lr_update_rate = reader.read_i32()?;
      let t = reader.read_f64()?;

      Ok(Self {
         dim,
         ws,
         epoch,
         min_count,
         neg,
         word_ngrams,
         loss,
         model_type,
         bucket,
         minn,
         maxn,
         lr_update_rate,
         t,
      })
   }
}
