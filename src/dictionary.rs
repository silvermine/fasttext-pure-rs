//! Word and label dictionary with subword hashing and tokenization.
//!
//! The dictionary is the second section of the fastText binary format,
//! following the model args header. It contains the word list, label list,
//! and metadata needed for subword hashing.

use std::collections::HashMap;
use std::io::Read;

use crate::args::ModelArgs;
use crate::error::{Error, Result};
use crate::io::BinaryReader;

const EOS: &str = "</s>";
const LABEL_PREFIX: &str = "__label__";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryType {
   Word = 0,
   Label = 1,
}

#[derive(Debug)]
struct Entry {
   word: String,
   count: i64,
   entry_type: EntryType,
   subwords: Vec<i32>,
}

/// The word/label dictionary loaded from a fastText model.
///
/// Handles tokenization, word lookup, subword n-gram hashing, and word
/// n-gram hashing. The hash function is FNV-1a (32-bit) and must be
/// bit-exact with the C++ implementation.
#[derive(Debug)]
pub(crate) struct Dictionary {
   args: ModelArgs,
   words: Vec<Entry>,
   word2int: HashMap<String, usize>,
   pub nwords: i32,
   #[allow(dead_code)]
   pub nlabels: i32,
   pruneidx_size: i64,
   pruneidx: HashMap<i32, i32>,
}

impl Dictionary {
   /// Load a dictionary from the binary format.
   pub fn load<R: Read>(reader: &mut BinaryReader<R>, args: &ModelArgs) -> Result<Self> {
      let size = reader.read_i32_as_usize()?;
      let nwords = reader.read_i32()?;
      let nlabels = reader.read_i32()?;
      let _ntokens = reader.read_i64()?;
      let pruneidx_size = reader.read_i64()?;

      let mut words = Vec::with_capacity(size);
      let mut word2int = HashMap::with_capacity(size);

      for i in 0..size {
         let word = reader.read_string()?;
         let count = reader.read_i64()?;
         let entry_type = match reader.read_u8()? {
            0 => EntryType::Word,
            1 => EntryType::Label,
            t => return Err(Error::InvalidModel(format!("unknown entry type: {t}"))),
         };

         word2int.insert(word.clone(), i);
         words.push(Entry {
            word,
            count,
            entry_type,
            subwords: Vec::new(),
         });
      }

      let mut pruneidx = HashMap::new();
      for _ in 0..pruneidx_size.max(0) {
         let first = reader.read_i32()?;
         let second = reader.read_i32()?;
         pruneidx.insert(first, second);
      }

      let mut dict = Self {
         args: args.clone(),
         words,
         word2int,
         nwords,
         nlabels,
         pruneidx_size,
         pruneidx,
      };

      dict.init_ngrams();

      Ok(dict)
   }

   /// FNV-1a 32-bit hash, bit-exact with the C++ fastText implementation.
   ///
   /// The C++ version sign-extends each byte before XOR:
   /// `h = h ^ uint32_t(int8_t(c))`. We replicate this via `byte as i8 as
   /// u32`.
   pub fn hash(word: &[u8]) -> u32 {
      let mut h: u32 = 2_166_136_261;
      for &byte in word {
         h ^= (byte as i8) as u32;
         h = h.wrapping_mul(16_777_619);
      }
      h
   }

   /// Tokenize input text into word IDs and subword hashes for prediction.
   ///
   /// Returns `(words, labels)` where `words` contains word IDs and
   /// subword hash indices, and `labels` contains any label IDs found in
   /// the input.
   pub fn get_line(&self, text: &str) -> (Vec<i32>, Vec<i32>) {
      let mut words = Vec::new();
      let mut labels = Vec::new();
      let mut word_hashes = Vec::new();

      for token in text.split_whitespace().chain(std::iter::once(EOS)) {
         let h = Self::hash(token.as_bytes());
         let wid = self.get_id(token);
         let entry_type = if wid < 0 {
            Self::get_type_by_token(token)
         } else {
            self.words[wid as usize].entry_type
         };

         if entry_type == EntryType::Word {
            self.add_subwords(&mut words, token, wid);
            word_hashes.push(h);
         } else if entry_type == EntryType::Label && wid >= 0 {
            labels.push(wid - self.nwords);
         }
      }

      self.add_word_ngrams(&mut words, &word_hashes, self.args.word_ngrams);
      (words, labels)
   }

   /// Get the label string for a label index.
   pub fn get_label(&self, lid: i32) -> &str {
      &self.words[(lid + self.nwords) as usize].word
   }

   /// Get the label counts in dictionary order (descending by count).
   ///
   /// Used to build the Huffman tree for hierarchical softmax prediction.
   pub fn get_label_counts(&self) -> Vec<i64> {
      self
         .words
         .iter()
         .filter(|e| e.entry_type == EntryType::Label)
         .map(|e| e.count)
         .collect()
   }

   /// Look up a word in the dictionary, returning its index or -1 if not
   /// found.
   fn get_id(&self, word: &str) -> i32 {
      self.word2int.get(word).map(|&id| id as i32).unwrap_or(-1)
   }

   /// Determine the entry type of a token based on the label prefix.
   fn get_type_by_token(token: &str) -> EntryType {
      if token.starts_with(LABEL_PREFIX) {
         EntryType::Label
      } else {
         EntryType::Word
      }
   }

   /// Initialize precomputed subword n-grams for all dictionary entries.
   fn init_ngrams(&mut self) {
      for i in 0..self.words.len() {
         self.words[i].subwords.clear();
         self.words[i].subwords.push(i as i32);
         if self.words[i].word != EOS {
            let mut subwords = Vec::new();
            self.compute_sub_words_with_markers(self.words[i].word.as_bytes(), &mut subwords);
            self.words[i].subwords.extend(subwords);
         }
      }
   }

   /// Compute character n-gram hashes for a word (with BOW/EOW markers).
   ///
   /// Iterates over UTF-8 character boundaries and hashes n-grams
   /// incrementally using the FNV-1a property: the hash of a prefix
   /// can be extended one byte at a time. This avoids allocating a
   /// temporary byte vector for each n-gram.
   fn compute_sub_words(&self, word: &[u8], ngrams: &mut Vec<i32>) {
      if self.args.maxn <= 0 {
         return;
      }

      let minn = self.args.minn as usize;
      let maxn = self.args.maxn as usize;

      for i in 0..word.len() {
         // Skip UTF-8 continuation bytes
         if (word[i] & 0xC0) == 0x80 {
            continue;
         }

         let mut h: u32 = 2_166_136_261; // FNV offset basis
         let mut j = i;
         let mut n: usize = 1;

         while j < word.len() && n <= maxn {
            // Hash the next byte
            h ^= (word[j] as i8) as u32;
            h = h.wrapping_mul(16_777_619);
            j += 1;

            // Consume remaining bytes of the current UTF-8 character
            while j < word.len() && (word[j] & 0xC0) == 0x80 {
               h ^= (word[j] as i8) as u32;
               h = h.wrapping_mul(16_777_619);
               j += 1;
            }

            // Exclude single-char n-grams that are BOW or EOW markers
            if n >= minn && !(n == 1 && (i == 0 || j == word.len())) {
               self.push_hash(ngrams, (h % self.args.bucket as u32) as i32);
            }

            n += 1;
         }
      }
   }

   /// Push a hashed subword index, respecting pruning state.
   ///
   /// - `pruneidx_size < 0` (e.g., -1): no pruning, push directly
   /// - `pruneidx_size == 0`: all subwords pruned, push nothing
   /// - `pruneidx_size > 0`: remap through `pruneidx`, skip if not found
   fn push_hash(&self, hashes: &mut Vec<i32>, id: i32) {
      if self.pruneidx_size == 0 || id < 0 {
         return;
      }
      if self.pruneidx_size > 0 {
         if let Some(&remapped) = self.pruneidx.get(&id) {
            hashes.push(self.nwords + remapped);
         }
         return;
      }
      // pruneidx_size < 0: no pruning, push raw hash index
      hashes.push(self.nwords + id);
   }

   /// Add word IDs and/or subword hashes for a token to the line.
   fn add_subwords(&self, line: &mut Vec<i32>, token: &str, wid: i32) {
      if wid < 0 {
         // Out of vocabulary: compute subwords on the fly
         if token != EOS {
            self.compute_sub_words_with_markers(token.as_bytes(), line);
         }
      } else if self.args.maxn <= 0 {
         // In vocabulary without subwords
         line.push(wid);
      } else {
         // In vocabulary with precomputed subwords
         line.extend_from_slice(&self.words[wid as usize].subwords);
      }
   }

   /// Compute subwords for a token by prepending "<" and appending ">"
   /// without allocating a new string. Uses a small stack buffer for
   /// tokens up to 256 bytes.
   fn compute_sub_words_with_markers(&self, token: &[u8], ngrams: &mut Vec<i32>) {
      let total_len = token.len() + 2; // "<" + token + ">"
      if total_len <= 258 {
         // Stack-allocated path for typical tokens
         let mut buf = [0u8; 258];
         buf[0] = b'<';
         buf[1..1 + token.len()].copy_from_slice(token);
         buf[1 + token.len()] = b'>';
         self.compute_sub_words(&buf[..total_len], ngrams);
      } else {
         // Heap-allocated fallback for very long tokens
         let mut word = Vec::with_capacity(total_len);
         word.push(b'<');
         word.extend_from_slice(token);
         word.push(b'>');
         self.compute_sub_words(&word, ngrams);
      }
   }

   /// Add word n-gram hashes (bigrams, trigrams, etc.) to the line.
   ///
   /// Uses 64-bit arithmetic with the word n-gram hash constant
   /// `116049371`, matching the C++ implementation.
   fn add_word_ngrams(&self, line: &mut Vec<i32>, hashes: &[u32], n: i32) {
      for i in 0..hashes.len() {
         let mut h = hashes[i] as u64;
         for hash in hashes
            .iter()
            .take(hashes.len().min(i + n as usize))
            .skip(i + 1)
         {
            h = h.wrapping_mul(116_049_371).wrapping_add(*hash as u64);
            self.push_hash(line, (h % (self.args.bucket as u64)) as i32);
         }
      }
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn hash_matches_cpp_fnv1a() {
      // Known FNV-1a values with C++ sign-extension behavior
      assert_eq!(Dictionary::hash(b"hello"), 1_335_831_723);
      assert_eq!(Dictionary::hash(b"world"), 933_488_787);
      assert_eq!(Dictionary::hash(b""), 2_166_136_261); // FNV offset basis
   }

   #[test]
   fn hash_empty_string_is_offset_basis() {
      assert_eq!(Dictionary::hash(b""), 0x811C_9DC5);
   }
}
