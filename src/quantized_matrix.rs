//! Product quantizer and quantized matrix for `.ftz` models.
//!
//! The product quantizer compresses weight matrices by splitting each
//! vector into sub-vectors and replacing each sub-vector with an 8-bit
//! index into a learned codebook of 256 centroids.

use std::io::Read;

use crate::error::{Error, Result};
use crate::io::BinaryReader;

/// Number of centroids per sub-quantizer (2^8 = 256 for 8-bit codes).
const KSUB: usize = 256;

/// Codebook-based vector compression using product quantization.
///
/// Each vector of dimension `dim` is split into `nsubq` sub-vectors.
/// Sub-vectors 0..nsubq-2 have dimension `dsub`; the last sub-vector
/// has dimension `lastdsub` (which may differ if `dim` is not evenly
/// divisible by `dsub`).
#[derive(Debug)]
pub(crate) struct ProductQuantizer {
   nsubq: usize,
   dsub: usize,
   lastdsub: usize,
   centroids: Vec<f32>,
}

impl ProductQuantizer {
   /// Load a product quantizer from the binary format.
   ///
   /// Reads `dim`, `nsubq`, `dsub`, `lastdsub` (all `i32`), followed by
   /// `dim * 256` centroid `f32` values.
   pub fn load<R: Read>(reader: &mut BinaryReader<R>) -> Result<Self> {
      let dim = reader.read_i32_as_usize()?;
      let nsubq = reader.read_i32_as_usize()?;
      let dsub = reader.read_i32_as_usize()?;
      let lastdsub = reader.read_i32_as_usize()?;
      let centroid_len = dim
         .checked_mul(KSUB)
         .ok_or_else(|| Error::InvalidModel("centroid size overflow".to_string()))?;
      let centroids = reader.read_f32_vec(centroid_len)?;
      Ok(Self {
         nsubq,
         dsub,
         lastdsub,
         centroids,
      })
   }

   /// Get a slice of centroid values for sub-quantizer `m`, code `i`.
   fn get_centroids(&self, m: usize, i: u8) -> &[f32] {
      let i = i as usize;
      if m == self.nsubq - 1 {
         let start = m * KSUB * self.dsub + i * self.lastdsub;
         &self.centroids[start..start + self.lastdsub]
      } else {
         let start = (m * KSUB + i) * self.dsub;
         &self.centroids[start..start + self.dsub]
      }
   }

   /// Reconstruct quantized row `t` and add it (scaled by `alpha`) to
   /// vector `x`.
   pub fn add_code(&self, x: &mut [f32], codes: &[u8], t: usize, alpha: f32) {
      let offset = self.nsubq * t;
      for m in 0..self.nsubq {
         let c = self.get_centroids(m, codes[m + offset]);
         let d = if m == self.nsubq - 1 {
            self.lastdsub
         } else {
            self.dsub
         };
         let base = m * self.dsub;
         for n in 0..d {
            x[base + n] += alpha * c[n];
         }
      }
   }

   /// Compute the dot product of vector `x` with quantized row `t`,
   /// scaled by `alpha`.
   pub fn mul_code(&self, x: &[f32], codes: &[u8], t: usize, alpha: f32) -> f32 {
      let mut res: f32 = 0.0;
      let offset = self.nsubq * t;
      for m in 0..self.nsubq {
         let c = self.get_centroids(m, codes[m + offset]);
         let d = if m == self.nsubq - 1 {
            self.lastdsub
         } else {
            self.dsub
         };
         let base = m * self.dsub;
         for n in 0..d {
            res += x[base + n] * c[n];
         }
      }
      res * alpha
   }
}

/// A quantized weight matrix wrapping a [`ProductQuantizer`] with
/// per-row codes and optional norm quantization.
#[derive(Debug)]
pub(crate) struct QuantMatrix {
   qnorm: bool,
   pub m: usize,
   #[allow(dead_code)]
   pub n: usize,
   codes: Vec<u8>,
   pq: ProductQuantizer,
   norm_codes: Vec<u8>,
   npq: Option<ProductQuantizer>,
}

impl QuantMatrix {
   /// Load a quantized matrix from the binary format.
   ///
   /// Format: `qnorm` (u8), `m` (i64), `n` (i64), `codesize` (i32),
   /// `codes` (u8 * codesize), main `ProductQuantizer`. If `qnorm` is
   /// true, also reads `norm_codes` (u8 * m) and a norm
   /// `ProductQuantizer`.
   pub fn load<R: Read>(reader: &mut BinaryReader<R>) -> Result<Self> {
      let qnorm = reader.read_u8()? != 0;
      let m = reader.read_i64_as_usize()?;
      let n = reader.read_i64_as_usize()?;
      let codesize = reader.read_i32_as_usize()?;
      let codes = reader.read_u8_vec(codesize)?;
      let pq = ProductQuantizer::load(reader)?;

      let (norm_codes, npq) = if qnorm {
         let nc = reader.read_u8_vec(m)?;
         let nq = ProductQuantizer::load(reader)?;
         (nc, Some(nq))
      } else {
         (Vec::new(), None)
      };

      Ok(Self {
         qnorm,
         m,
         n,
         codes,
         pq,
         norm_codes,
         npq,
      })
   }

   /// Add quantized row `t` to vector `x`, applying norm scaling if
   /// present.
   pub fn add_to_vector(&self, x: &mut [f32], t: usize) -> Result<()> {
      let norm = self.get_norm(t)?;
      self.pq.add_code(x, &self.codes, t, norm);
      Ok(())
   }

   /// Compute the dot product of `vec` with quantized row `i`, applying
   /// norm scaling if present.
   pub fn dot_row(&self, vec: &[f32], i: usize) -> Result<f32> {
      let norm = self.get_norm(i)?;
      Ok(self.pq.mul_code(vec, &self.codes, i, norm))
   }

   /// Get the norm scaling factor for row `t`.
   fn get_norm(&self, t: usize) -> Result<f32> {
      if self.qnorm {
         let npq = self.npq.as_ref().ok_or_else(|| {
            Error::InvalidModel("qnorm is true but norm quantizer is missing".to_string())
         })?;
         Ok(npq.get_centroids(0, self.norm_codes[t])[0])
      } else {
         Ok(1.0)
      }
   }
}
