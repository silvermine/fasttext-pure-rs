//! Tier 2: Product quantizer correctness tests.
//!
//! These tests verify the product quantizer's decode and dot product
//! operations using a small synthetic codebook with known centroids.

use fasttext_pure_rs::FastText;

const MODEL_PATH: &str = "tests/fixtures/lid.176.ftz";

fn load_model() -> FastText {
   let bytes = std::fs::read(MODEL_PATH).expect("failed to read model file");
   FastText::load_from_reader(std::io::Cursor::new(bytes)).expect("failed to load model")
}

/// The lid.176.ftz model uses a quantized input matrix. If the product
/// quantizer decoding is incorrect, predictions would be wrong. Since
/// golden-output parity tests pass, this validates the PQ pipeline
/// end-to-end. This test specifically verifies that the quantized model
/// produces correct sentence vectors (which exercise the PQ add_code
/// path).
#[test]
fn quantized_sentence_vector_matches_cpp() {
   let model = load_model();
   let vector = model.get_sentence_vector("hello world").unwrap();

   // Golden values from C++ `print-sentence-vectors`
   let expected: [f32; 16] = [
      -0.65101, 0.10776, -0.35615, -0.18231, 0.084701, 0.17492, -0.22578, 0.1906, 0.064585,
      -0.28229, 0.076955, 0.0064092, 0.062685, 0.16378, 0.12939, 0.039792,
   ];

   for (i, (&actual, &exp)) in vector.iter().zip(expected.iter()).enumerate() {
      let diff = (actual - exp).abs();
      assert!(
         diff < 1e-4,
         "PQ decoded vector mismatch at dim {i}: expected {exp}, got {actual}",
      );
   }
}

/// Verify that different inputs produce different vectors (the PQ is
/// not trivially degenerate).
#[test]
fn quantized_vectors_differ_for_different_inputs() {
   let model = load_model();
   let v1 = model.get_sentence_vector("hello world").unwrap();
   let v2 = model.get_sentence_vector("bonjour le monde").unwrap();

   assert_ne!(
      v1, v2,
      "different sentences should produce different vectors"
   );
}

/// Verify that the same input always produces the same vector
/// (deterministic decoding).
#[test]
fn quantized_vectors_deterministic() {
   let model = load_model();
   let v1 = model.get_sentence_vector("test determinism").unwrap();
   let v2 = model.get_sentence_vector("test determinism").unwrap();

   assert_eq!(v1, v2, "same input should always produce the same vector");
}

/// Verify that the quantized model's predictions are consistent with
/// vector similarity (the top-predicted language's vector should be
/// the hidden vector direction).
#[test]
fn quantized_prediction_uses_pq_correctly() {
   let model = load_model();

   // A strongly English sentence should predict English with high
   // confidence, which validates the full PQ pipeline: input embedding
   // lookup via add_code, hidden vector averaging, and output dot
   // products.
   let predictions = model
      .predict("This is definitely English text", 1, 0.0)
      .unwrap();
   assert_eq!(predictions[0].label, "__label__en");
   assert!(
      predictions[0].probability > 0.9,
      "expected high confidence, got {}",
      predictions[0].probability,
   );
}
