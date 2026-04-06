//! Low-level binary format reader utilities for fastText model files.

use std::io::Read;

use crate::error::{Error, Result};

/// Maximum single allocation size (1 GB). Prevents out-of-memory
/// crashes from malformed model files claiming absurdly large
/// dimensions.
const MAX_ALLOC_BYTES: usize = 1_073_741_824;

/// Maximum length of a null-terminated string (1 MiB). fastText
/// dictionary strings are short words/labels, so this is extremely
/// generous. Prevents unbounded memory growth if a malformed model
/// file is missing a null terminator.
const MAX_STRING_BYTES: usize = 1_048_576;

/// Magic number identifying a fastText binary model file.
pub(crate) const FASTTEXT_MAGIC: i32 = 793_712_314;

/// The only supported binary format version.
pub(crate) const FASTTEXT_VERSION: i32 = 12;

/// A sequential reader for little-endian binary data from a fastText model
/// file.
pub(crate) struct BinaryReader<R: Read> {
   reader: R,
}

impl<R: Read> BinaryReader<R> {
   /// Wrap a `Read` source in a `BinaryReader`.
   pub fn new(reader: R) -> Self {
      Self { reader }
   }

   /// Validate the magic number and version at the start of a model file.
   pub fn validate_header(&mut self) -> Result<()> {
      let magic = self.read_i32()?;
      if magic != FASTTEXT_MAGIC {
         return Err(Error::InvalidModel(format!(
            "invalid magic number: expected {FASTTEXT_MAGIC}, got {magic}"
         )));
      }

      let version = self.read_i32()?;
      if version != FASTTEXT_VERSION {
         return Err(Error::UnsupportedVersion {
            expected: FASTTEXT_VERSION,
            actual: version,
         });
      }

      Ok(())
   }

   /// Read a single unsigned byte.
   pub fn read_u8(&mut self) -> Result<u8> {
      let mut buf = [0u8; 1];
      self.reader.read_exact(&mut buf)?;
      Ok(buf[0])
   }

   /// Read a little-endian `i32`.
   pub fn read_i32(&mut self) -> Result<i32> {
      let mut buf = [0u8; 4];
      self.reader.read_exact(&mut buf)?;
      Ok(i32::from_le_bytes(buf))
   }

   /// Read a little-endian `i64`.
   pub fn read_i64(&mut self) -> Result<i64> {
      let mut buf = [0u8; 8];
      self.reader.read_exact(&mut buf)?;
      Ok(i64::from_le_bytes(buf))
   }

   /// Read a little-endian `i64` and convert to `usize`, returning an
   /// error if the value is negative.
   pub fn read_i64_as_usize(&mut self) -> Result<usize> {
      let raw = self.read_i64()?;
      usize::try_from(raw).map_err(|_| Error::InvalidModel(format!("negative i64 value: {raw}")))
   }

   /// Read a little-endian `i32` and convert to `usize`, returning an
   /// error if the value is negative.
   pub fn read_i32_as_usize(&mut self) -> Result<usize> {
      let raw = self.read_i32()?;
      usize::try_from(raw).map_err(|_| Error::InvalidModel(format!("negative i32 value: {raw}")))
   }

   /// Read a little-endian `f64`.
   pub fn read_f64(&mut self) -> Result<f64> {
      let mut buf = [0u8; 8];
      self.reader.read_exact(&mut buf)?;
      Ok(f64::from_le_bytes(buf))
   }

   /// Read a null-terminated string.
   ///
   /// Rejects strings longer than [`MAX_STRING_BYTES`] to prevent
   /// unbounded memory growth from malformed model files missing a
   /// null terminator.
   pub fn read_string(&mut self) -> Result<String> {
      let mut bytes = Vec::new();
      loop {
         let b = self.read_u8()?;
         if b == 0 {
            break;
         }
         bytes.push(b);
         if bytes.len() > MAX_STRING_BYTES {
            return Err(Error::InvalidModel(format!(
               "string exceeds {MAX_STRING_BYTES} byte limit"
            )));
         }
      }
      String::from_utf8(bytes)
         .map_err(|e| Error::InvalidModel(format!("invalid UTF-8 in dictionary string: {e}")))
   }

   /// Read `len` bytes into a `Vec<u8>`.
   ///
   /// Rejects allocations larger than [`MAX_ALLOC_BYTES`] to prevent
   /// out-of-memory crashes from malformed model files.
   pub fn read_u8_vec(&mut self, len: usize) -> Result<Vec<u8>> {
      if len > MAX_ALLOC_BYTES {
         return Err(Error::InvalidModel(format!(
            "data too large: {len} bytes exceeds {MAX_ALLOC_BYTES} byte limit"
         )));
      }
      let mut buf = vec![0u8; len];
      self.reader.read_exact(&mut buf)?;
      Ok(buf)
   }

   /// Read `len` little-endian `f32` values into a `Vec<f32>`.
   ///
   /// Rejects allocations larger than [`MAX_ALLOC_BYTES`] to prevent
   /// out-of-memory crashes from malformed model files.
   pub fn read_f32_vec(&mut self, len: usize) -> Result<Vec<f32>> {
      let byte_len = len
         .checked_mul(4)
         .ok_or_else(|| Error::InvalidModel("matrix size overflow".to_string()))?;
      if byte_len > MAX_ALLOC_BYTES {
         return Err(Error::InvalidModel(format!(
            "matrix data too large: {byte_len} bytes exceeds {MAX_ALLOC_BYTES} byte limit"
         )));
      }
      let mut buf = vec![0u8; byte_len];
      self.reader.read_exact(&mut buf)?;
      let data = buf
         .chunks_exact(4)
         .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk is exactly 4 bytes")))
         .collect();
      Ok(data)
   }
}

#[cfg(test)]
mod tests {
   use super::*;
   use std::io::Cursor;

   #[test]
   fn read_i32_little_endian() {
      let data: Vec<u8> = vec![0x2A, 0x00, 0x00, 0x00];
      let mut reader = BinaryReader::new(Cursor::new(data));
      assert_eq!(reader.read_i32().unwrap(), 42);
   }

   #[test]
   fn read_null_terminated_string() {
      let data: Vec<u8> = vec![b'h', b'i', 0];
      let mut reader = BinaryReader::new(Cursor::new(data));
      assert_eq!(reader.read_string().unwrap(), "hi");
   }

   #[test]
   fn validate_header_rejects_bad_magic() {
      let mut data = Vec::new();
      data.extend_from_slice(&0i32.to_le_bytes());
      data.extend_from_slice(&12i32.to_le_bytes());
      let mut reader = BinaryReader::new(Cursor::new(data));
      assert!(reader.validate_header().is_err());
   }

   #[test]
   fn validate_header_rejects_bad_version() {
      let mut data = Vec::new();
      data.extend_from_slice(&FASTTEXT_MAGIC.to_le_bytes());
      data.extend_from_slice(&11i32.to_le_bytes());
      let mut reader = BinaryReader::new(Cursor::new(data));
      let err = reader.validate_header().unwrap_err();
      assert!(matches!(
         err,
         Error::UnsupportedVersion {
            expected: 12,
            actual: 11
         }
      ));
   }

   #[test]
   fn validate_header_accepts_valid() {
      let mut data = Vec::new();
      data.extend_from_slice(&FASTTEXT_MAGIC.to_le_bytes());
      data.extend_from_slice(&FASTTEXT_VERSION.to_le_bytes());
      let mut reader = BinaryReader::new(Cursor::new(data));
      assert!(reader.validate_header().is_ok());
   }
}
