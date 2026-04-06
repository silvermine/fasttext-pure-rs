use fasttext_pure_rs::FastText;

const MODEL_PATH: &str = "tests/fixtures/lid.176.ftz";

#[cfg(feature = "std")]
#[test]
fn load_ftz_model() {
   let model = FastText::load(MODEL_PATH);
   assert!(model.is_ok(), "failed to load model: {:?}", model.err());
}

#[test]
fn load_from_reader() {
   let bytes = std::fs::read(MODEL_PATH).unwrap();
   let model = FastText::load_from_reader(std::io::Cursor::new(bytes));
   assert!(
      model.is_ok(),
      "failed to load model from reader: {:?}",
      model.err()
   );
}

#[test]
fn reject_invalid_magic() {
   let data = vec![0u8; 100];
   let result = FastText::load_from_reader(std::io::Cursor::new(data));
   assert!(result.is_err());
   let err = result.unwrap_err();
   assert!(
      matches!(err, fasttext_pure_rs::Error::InvalidModel(_)),
      "expected InvalidModel, got: {err:?}"
   );
}

#[test]
fn reject_truncated_model() {
   // Valid header but truncated after args — should fail during
   // dictionary loading, not panic
   let bytes = std::fs::read(MODEL_PATH).unwrap();
   let truncated = &bytes[..100]; // just magic + version + partial args
   let result = FastText::load_from_reader(std::io::Cursor::new(truncated));
   assert!(result.is_err(), "truncated model should fail to load");
   assert!(
      matches!(result.unwrap_err(), fasttext_pure_rs::Error::Io(_)),
      "expected Io error for truncated file"
   );
}

#[test]
fn reject_corrupt_args() {
   // Valid magic and version, then garbage — should fail during args
   // parsing or dictionary loading
   let mut data = Vec::new();
   data.extend_from_slice(&793_712_314i32.to_le_bytes());
   data.extend_from_slice(&12i32.to_le_bytes());
   data.extend_from_slice(&[0xFF; 200]); // garbage args + dictionary

   let result = FastText::load_from_reader(std::io::Cursor::new(data));
   assert!(result.is_err(), "corrupt model should fail to load");
}

#[cfg(feature = "std")]
#[test]
fn dim_accessor() {
   let model = FastText::load(MODEL_PATH).unwrap();
   assert_eq!(model.dim(), 16, "lid.176.ftz should have dim=16");
}

#[test]
fn reject_unsupported_version() {
   let mut data = Vec::new();
   data.extend_from_slice(&793_712_314i32.to_le_bytes()); // valid magic
   data.extend_from_slice(&11i32.to_le_bytes()); // unsupported version
   data.extend_from_slice(&[0u8; 100]); // filler

   let result = FastText::load_from_reader(std::io::Cursor::new(data));
   assert!(result.is_err());
   let err = result.unwrap_err();
   assert!(
      matches!(
         err,
         fasttext_pure_rs::Error::UnsupportedVersion {
            expected: 12,
            actual: 11
         }
      ),
      "expected UnsupportedVersion, got: {err:?}"
   );
}
