//! Golden-output parity tests for unquantized `.bin` model format.
//!
//! These tests load `lid.176.bin` (~126 MB, tracked via Git LFS) and
//! verify prediction parity with the C++ `fasttext` binary. Tests
//! gracefully skip when the LFS object is not available (e.g., shallow
//! clone without LFS checkout).
//!
//! All test sentences are real-world text sourced from jw.org.

use std::io::Read;

use fasttext_pure_rs::FastText;

const BIN_MODEL_PATH: &str = "tests/fixtures/lid.176.bin";

/// The fastText binary magic number as little-endian bytes.
/// Canonical source: `FASTTEXT_MAGIC` in `src/io.rs` (pub(crate), so not
/// directly accessible from integration tests).
const FASTTEXT_MAGIC_LE: [u8; 4] = 793_712_314i32.to_le_bytes();

const PROB_REL_TOLERANCE: f32 = 0.02;

/// Check whether the file at `path` begins with the fastText magic
/// number. Returns `false` if the file is missing, unreadable, or
/// contains an LFS pointer instead of real model data.
fn has_valid_magic(path: &str) -> bool {
   let mut f = match std::fs::File::open(path) {
      Ok(f) => f,
      Err(_) => return false,
   };
   let mut buf = [0u8; 4];
   f.read_exact(&mut buf).is_ok() && buf == FASTTEXT_MAGIC_LE
}

fn try_load_bin_model() -> Option<FastText> {
   if !has_valid_magic(BIN_MODEL_PATH) {
      eprintln!("SKIP: {BIN_MODEL_PATH} not found or not a valid model (LFS not configured?)");
      return None;
   }

   Some(FastText::load(BIN_MODEL_PATH).expect("failed to load .bin model"))
}

/// Define a test that gracefully skips when the Git LFS model fixture
/// is not available (missing file or LFS pointer instead of real data).
macro_rules! lfs_test {
   ($name:ident, $body:expr) => {
      #[test]
      fn $name() {
         let Some(model) = try_load_bin_model() else {
            return;
         };
         #[allow(clippy::redundant_closure_call)]
         ($body)(&model);
      }
   };
}

fn assert_top1(model: &FastText, text: &str, expected_label: &str, expected_prob: f32) {
   let predictions = model.predict(text, 1, 0.0).unwrap();
   assert_eq!(
      predictions.len(),
      1,
      "expected 1 prediction for {expected_label}"
   );
   assert_eq!(
      predictions[0].label, expected_label,
      "wrong label for {expected_label}"
   );
   let actual = predictions[0].probability;
   let diff = (actual - expected_prob).abs();
   let tolerance = expected_prob * PROB_REL_TOLERANCE;
   assert!(
      diff <= tolerance,
      "prob mismatch for {expected_label}: expected {expected_prob}, got {actual}",
   );
}

lfs_test!(load_bin_model, |model: &FastText| {
   assert_eq!(model.dim(), 16);
});

lfs_test!(golden_bin_en, |model: &FastText| {
   assert_top1(
      model,
      "The Bible is a collection of 66 sacred books written over a period of approximately 1,600 years",
      "__label__en",
      0.954254,
   );
});

lfs_test!(golden_bin_fr, |model: &FastText| {
   assert_top1(
      model,
      "La Bible est un recueil de 66 livres sacr\u{00e9}s qui contient un message divin adress\u{00e9} aux humains",
      "__label__fr",
      0.884887,
   );
});

lfs_test!(golden_bin_de, |model: &FastText| {
   assert_top1(
      model,
      "Hier wird gezeigt, warum die Bibel vertrauensw\u{00fc}rdig ist und wie sie uns im Alltag helfen kann",
      "__label__de",
      0.999802,
   );
});

lfs_test!(golden_bin_es, |model: &FastText| {
   assert_top1(
      model,
      "La Biblia est\u{00e1} compuesta por 66 libros sagrados y se escribi\u{00f3} durante un per\u{00ed}odo de unos 1.600 a\u{00f1}os",
      "__label__es",
      0.992926,
   );
});

lfs_test!(golden_bin_ja, |model: &FastText| {
   assert_top1(
      model,
      "\u{3053}\u{306e}\u{30bb}\u{30af}\u{30b7}\u{30e7}\u{30f3}\u{306f}\u{ff0c}\u{8056}\u{66f8}\u{3092}\u{4fe1}\u{983c}\u{3067}\u{304d}\u{308b}\u{306e}\u{306f}\u{306a}\u{305c}\u{304b}\u{ff0c}\u{8056}\u{66f8}\u{3092}\u{6700}\u{5927}\u{9650}\u{306b}\u{6d3b}\u{7528}\u{3059}\u{308b}\u{306b}\u{306f}\u{3069}\u{3046}\u{3059}\u{308c}\u{3070}\u{3088}\u{3044}\u{304b}\u{ff0c}\u{8056}\u{66f8}\u{304c}\u{3069}\u{308c}\u{307b}\u{3069}\u{5f79}\u{7acb}\u{3064}\u{304b}\u{3092}\u{53d6}\u{308a}\u{4e0a}\u{3052}\u{3066}\u{3044}\u{307e}\u{3059}",
      "__label__ja",
      0.998826,
   );
});

lfs_test!(golden_bin_zh, |model: &FastText| {
   assert_top1(
      model,
      "\u{5723}\u{7ecf}\u{503c}\u{5f97}\u{4e00}\u{770b}\u{5417}\u{ff1f}\u{5723}\u{7ecf}\u{5bf9}\u{6211}\u{6709}\u{5e2e}\u{52a9}\u{5417}\u{ff1f}\u{8bf7}\u{770b}\u{770b}\u{8fd9}\u{4e2a}\u{4e13}\u{680f}\u{7684}\u{8d44}\u{6599}\u{ff0c}\u{4f60}\u{4f1a}\u{53d1}\u{73b0}\u{9605}\u{8bfb}\u{5723}\u{7ecf}\u{7684}\u{786e}\u{5927}\u{6709}\u{4ef7}\u{503c}",
      "__label__zh",
      0.934139,
   );
});

lfs_test!(golden_bin_ko, |model: &FastText| {
   assert_top1(
      model,
      "\u{c774} \u{c139}\u{c158}\u{c5d0} \u{b098}\u{c624}\u{b294} \u{c8fc}\u{c81c}\u{b4e4}\u{c744} \u{c0b4}\u{d3b4}\u{bcf4}\u{ba74} \u{c131}\u{acbd}\u{c744} \u{c2e0}\u{b8b0}\u{d560} \u{c218} \u{c788}\u{b294} \u{c774}\u{c720}\u{ac00} \u{bb34}\u{c5c7}\u{c778}\u{c9c0} \u{c5b4}\u{b5bb}\u{ac8c} \u{c131}\u{acbd}\u{c5d0}\u{c11c} \u{cd5c}\u{b300}\u{c758} \u{c720}\u{c775}\u{c744} \u{c5bb}\u{c744} \u{c218} \u{c788}\u{b294}\u{c9c0} \u{c54c} \u{c218} \u{c788}\u{c2b5}\u{b2c8}\u{b2e4}",
      "__label__ko",
      1.00007,
   );
});

lfs_test!(golden_bin_hi, |model: &FastText| {
   assert_top1(
      model,
      "\u{0907}\u{0938} \u{092d}\u{093e}\u{0917} \u{092e}\u{0947}\u{0902} \u{0906}\u{092a} \u{0926}\u{0947}\u{0916}\u{0947}\u{0902}\u{0917}\u{0947} \u{0915}\u{093f} \u{0906}\u{092a} \u{092a}\u{0935}\u{093f}\u{0924}\u{094d}\u{0930} \u{0936}\u{093e}\u{0938}\u{094d}\u{0924}\u{094d}\u{0930} \u{092a}\u{0930} \u{0915}\u{094d}\u{092f}\u{094b}\u{0902} \u{092f}\u{0915}\u{0940}\u{0928} \u{0915}\u{0930} \u{0938}\u{0915}\u{0924}\u{0947} \u{0939}\u{0948}\u{0902} \u{0914}\u{0930} \u{092a}\u{0935}\u{093f}\u{0924}\u{094d}\u{0930} \u{0936}\u{093e}\u{0938}\u{094d}\u{0924}\u{094d}\u{0930} \u{092e}\u{0947}\u{0902} \u{0926}\u{0940} \u{0938}\u{0932}\u{093e}\u{0939} \u{0906}\u{091c} \u{0939}\u{092e}\u{093e}\u{0930}\u{0947} \u{0932}\u{093f}\u{090f} \u{0915}\u{093f}\u{0924}\u{0928}\u{0940} \u{092b}\u{093e}\u{092f}\u{0926}\u{0947}\u{092e}\u{0902}\u{0926} \u{0939}\u{0948}",
      "__label__hi",
      0.999598,
   );
});

lfs_test!(golden_bin_ar, |model: &FastText| {
   assert_top1(
      model,
      "\u{0627}\u{0644}\u{0643}\u{062a}\u{0627}\u{0628} \u{0627}\u{0644}\u{0645}\u{0642}\u{062f}\u{0633} \u{0647}\u{0648} \u{0645}\u{062c}\u{0645}\u{0648}\u{0639}\u{0629} \u{0645}\u{0646} \u{0666}\u{0666} \u{0633}\u{0641}\u{0631}\u{064b}\u{0627} \u{0643}\u{064f}\u{062a}\u{0628}\u{062a} \u{0639}\u{0644}\u{0649} \u{0645}\u{062f}\u{0649} \u{0666}\u{0660}\u{0660} \u{0633}\u{0646}\u{0629} \u{062a}\u{0642}\u{0631}\u{064a}\u{0628}\u{064b}\u{0627} \u{0648}\u{064a}\u{062d}\u{062a}\u{0648}\u{064a} \u{0639}\u{0644}\u{0649} \u{0631}\u{0633}\u{0627}\u{0644}\u{0629} \u{0627}\u{0644}\u{0644}\u{0647} \u{0644}\u{0644}\u{0628}\u{0634}\u{0631}",
      "__label__ar",
      0.988783,
   );
});

lfs_test!(golden_bin_vi, |model: &FastText| {
   assert_top1(
      model,
      "Nh\u{1eef}ng ch\u{1ee7} \u{0111}\u{1ec1} trong ph\u{1ea7}n n\u{00e0}y cho th\u{1ea5}y l\u{00fd} do c\u{00f3} th\u{1ec3} tin c\u{1ead}y Kinh Th\u{00e1}nh v\u{00e0} Kinh Th\u{00e1}nh th\u{1ead}t s\u{1ef1} thi\u{1ebf}t th\u{1ef1}c nh\u{01b0} th\u{1ebf} n\u{00e0}o",
      "__label__vi",
      1.00006,
   );
});

lfs_test!(golden_bin_sw, |model: &FastText| {
   assert_top1(
      model,
      "Tafuta majibu ya maswali ya Biblia na upate msaada kwa ajili ya familia yako",
      "__label__sw",
      0.797634,
   );
});

lfs_test!(golden_bin_ka, |model: &FastText| {
   assert_top1(
      model,
      "\u{10d0}\u{10db} \u{10d2}\u{10d0}\u{10dc}\u{10e7}\u{10dd}\u{10e4}\u{10d8}\u{10da}\u{10d4}\u{10d1}\u{10d0}\u{10e8}\u{10d8} \u{10d2}\u{10d0}\u{10dc}\u{10d7}\u{10d0}\u{10d5}\u{10e1}\u{10d4}\u{10d1}\u{10e3}\u{10da}\u{10d8} \u{10e1}\u{10e2}\u{10d0}\u{10e2}\u{10d8}\u{10d4}\u{10d1}\u{10d8}\u{10d3}\u{10d0}\u{10dc} \u{10d2}\u{10d0}\u{10d8}\u{10d2}\u{10d4}\u{10d1}\u{10d7} \u{10e0}\u{10d0}\u{10e2}\u{10dd}\u{10db} \u{10e8}\u{10d4}\u{10d2}\u{10d8}\u{10eb}\u{10da}\u{10d8}\u{10d0}\u{10d7} \u{10d4}\u{10dc}\u{10d3}\u{10dd}\u{10d7} \u{10d1}\u{10d8}\u{10d1}\u{10da}\u{10d8}\u{10d0}\u{10e1} \u{10d3}\u{10d0} \u{10e0}\u{10d0}\u{10db}\u{10d3}\u{10d4}\u{10dc}\u{10d0}\u{10d3} \u{10de}\u{10e0}\u{10d0}\u{10e5}\u{10e2}\u{10d8}\u{10d9}\u{10e3}\u{10da}\u{10d8}\u{10d0} \u{10db}\u{10d0}\u{10e1}\u{10e8}\u{10d8} \u{10e9}\u{10d0}\u{10ec}\u{10d4}\u{10e0}\u{10d8}\u{10da}\u{10d8} \u{10e0}\u{10e9}\u{10d4}\u{10d5}\u{10d4}\u{10d1}\u{10d8}",
      "__label__ka",
      0.999976,
   );
});

lfs_test!(golden_bin_hy, |model: &FastText| {
   assert_top1(
      model,
      "\u{0533}\u{057f}\u{0565}\u{0584} \u{0561}\u{057d}\u{057f}\u{057e}\u{0561}\u{056e}\u{0561}\u{0577}\u{0576}\u{0579}\u{0575}\u{0561}\u{0576} \u{0570}\u{0561}\u{0580}\u{0581}\u{0565}\u{0580}\u{056b} \u{057a}\u{0561}\u{057f}\u{0561}\u{057d}\u{056d}\u{0561}\u{0576}\u{0576}\u{0565}\u{0580}\u{0568} \u{0587} \u{0563}\u{0578}\u{0580}\u{056e}\u{0576}\u{0561}\u{056f}\u{0561}\u{0576} \u{056d}\u{0578}\u{0580}\u{0570}\u{0578}\u{0582}\u{0580}\u{0564}\u{0576}\u{0565}\u{0580} \u{057d}\u{057f}\u{0561}\u{0581}\u{0565}\u{0584} \u{0568}\u{0576}\u{057f}\u{0561}\u{0576}\u{056b}\u{0584}\u{056b} \u{057e}\u{0565}\u{0580}\u{0561}\u{0562}\u{0565}\u{0580}\u{0575}\u{0561}\u{056c}",
      "__label__hy",
      1.00008,
   );
});

lfs_test!(bin_predictions_sorted_descending, |model: &FastText| {
   let predictions = model.predict("hello world", 5, 0.0).unwrap();
   for i in 1..predictions.len() {
      assert!(
         predictions[i - 1].probability >= predictions[i].probability,
         "not sorted: {} < {}",
         predictions[i - 1].probability,
         predictions[i].probability,
      );
   }
});

lfs_test!(bin_matches_ftz_top1, |model: &FastText| {
   let ftz = FastText::load("tests/fixtures/lid.176.ftz").expect("failed to load .ftz");
   let sentences = [
      "The Bible is a collection of 66 sacred books written over a period of approximately 1,600 years",
      "La Bible est un recueil de 66 livres sacr\u{00e9}s qui contient un message divin adress\u{00e9} aux humains",
      "Hier wird gezeigt, warum die Bibel vertrauensw\u{00fc}rdig ist und wie sie uns im Alltag helfen kann",
      "\u{3053}\u{306e}\u{30bb}\u{30af}\u{30b7}\u{30e7}\u{30f3}\u{306f}\u{ff0c}\u{8056}\u{66f8}\u{3092}\u{4fe1}\u{983c}\u{3067}\u{304d}\u{308b}\u{306e}\u{306f}\u{306a}\u{305c}\u{304b}",
   ];
   for text in &sentences {
      let bin_pred = model.predict(text, 1, 0.0).unwrap();
      let ftz_pred = ftz.predict(text, 1, 0.0).unwrap();
      assert_eq!(
         bin_pred[0].label, ftz_pred[0].label,
         ".bin/.ftz disagree for text"
      );
   }
});

lfs_test!(golden_bin_sentence_vector, |model: &FastText| {
   let vector = model.get_sentence_vector("hello world").unwrap();
   assert_eq!(vector.len(), 16);
   let expected: [f32; 16] = [
      -0.085922, 0.00087653, -0.069362, -0.03359, 0.082807, -0.0084899, -0.037258, 0.050734,
      0.001838, -0.032356, -0.010034, -0.034641, -0.0051592, 0.0088575, 0.035956, -0.015707,
   ];
   for (i, (&actual, &exp)) in vector.iter().zip(expected.iter()).enumerate() {
      let diff = (actual - exp).abs();
      assert!(diff < 1e-4, "dim {i}: expected {exp}, got {actual}");
   }
});
