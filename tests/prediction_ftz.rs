//! Tier 1: Golden-output parity tests against the C++ `fasttext` binary.
//!
//! All test sentences are real-world text sourced from jw.org, covering
//! 35 languages across Latin, Cyrillic, CJK, Indic, Arabic, and other
//! scripts. Each test verifies that our Rust implementation produces the
//! same top-1 label and a probability within tolerance of the C++ output.

use fasttext_pure_rs::FastText;

const MODEL_PATH: &str = "tests/fixtures/lid.176.ftz";

/// Relative tolerance for probability comparison. Hierarchical softmax
/// predictions can diverge slightly due to sigmoid implementation
/// differences (C++ uses a 512-entry lookup table, we compute directly).
const PROB_REL_TOLERANCE: f32 = 0.02;

fn load_model() -> FastText {
   FastText::load(MODEL_PATH).expect("failed to load test model")
}

fn assert_top1(model: &FastText, text: &str, expected_label: &str, expected_prob: f32) {
   let predictions = model.predict(text, 1, 0.0).unwrap();
   assert_eq!(
      predictions.len(),
      1,
      "expected exactly 1 prediction for {expected_label}"
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
      "probability mismatch for {expected_label}: expected {expected_prob}, got {actual} \
       (diff {diff} > tolerance {tolerance})",
   );
}

// --- Golden parity: European (Latin script) ---

#[test]
fn golden_ftz_en() {
   let model = load_model();
   assert_top1(
      &model,
      "The Bible is a collection of 66 sacred books written over a period of approximately 1,600 years",
      "__label__en",
      0.967531,
   );
}

#[test]
fn golden_ftz_fr() {
   let model = load_model();
   assert_top1(
      &model,
      "La Bible est un recueil de 66 livres sacr\u{00e9}s qui contient un message divin adress\u{00e9} aux humains",
      "__label__fr",
      0.899228,
   );
}

#[test]
fn golden_ftz_de() {
   let model = load_model();
   assert_top1(
      &model,
      "Hier wird gezeigt, warum die Bibel vertrauensw\u{00fc}rdig ist und wie sie uns im Alltag helfen kann",
      "__label__de",
      0.999856,
   );
}

#[test]
fn golden_ftz_es() {
   let model = load_model();
   assert_top1(
      &model,
      "La Biblia est\u{00e1} compuesta por 66 libros sagrados y se escribi\u{00f3} durante un per\u{00ed}odo de unos 1.600 a\u{00f1}os",
      "__label__es",
      0.97559,
   );
}

#[test]
fn golden_ftz_it() {
   let model = load_model();
   assert_top1(
      &model,
      "Gli argomenti trattati in questa pagina evidenziano quanto la Bibbia sia affidabile e quanto siano utili i suoi consigli",
      "__label__it",
      0.991111,
   );
}

#[test]
fn golden_ftz_nl() {
   let model = load_model();
   assert_top1(
      &model,
      "De Bijbel is een verzameling van 66 heilige boeken, geschreven in een periode van ongeveer 1600 jaar",
      "__label__nl",
      0.99724,
   );
}

#[test]
fn golden_ftz_pt() {
   let model = load_model();
   assert_top1(
      &model,
      "A B\u{00ed}blia \u{00e9} um conjunto de 66 livros sagrados. Ela foi escrita num per\u{00ed}odo de mais ou menos 1.600 anos",
      "__label__pt",
      0.946129,
   );
}

#[test]
fn golden_ftz_sv() {
   let model = load_model();
   assert_top1(
      &model,
      "Artiklarna i den h\u{00e4}r sektionen visar varf\u{00f6}r man kan lita p\u{00e5} Bibeln och vilket stort v\u{00e4}rde den har f\u{00f6}r oss i dag",
      "__label__sv",
      0.970969,
   );
}

#[test]
fn golden_ftz_fi() {
   let model = load_model();
   assert_top1(
      &model,
      "T\u{00e4}st\u{00e4} osiosta k\u{00e4}y ilmi miksi voit luottaa Raamattuun ja kuinka k\u{00e4}yt\u{00e4}nn\u{00f6}llinen kirja Raamattu on",
      "__label__fi",
      0.998026,
   );
}

#[test]
fn golden_ftz_pl() {
   let model = load_model();
   assert_top1(
      &model,
      "Biblia sk\u{0142}ada si\u{0119} z 66 \u{015b}wi\u{0119}tych ksi\u{0105}g. Spisywano j\u{0105} przez jakie\u{015b} 1600 lat",
      "__label__pl",
      0.999886,
   );
}

#[test]
fn golden_ftz_id() {
   let model = load_model();
   assert_top1(
      &model,
      "Alkitab adalah kumpulan dari 66 buku suci yang ditulis dalam jangka waktu kira-kira 1.600 tahun",
      "__label__id",
      0.939225,
   );
}

#[test]
fn golden_ftz_tl() {
   let model = load_model();
   assert_top1(
      &model,
      "Ipapakita ng mga paksa sa seksiyong ito kung bakit ka makapagtitiwala sa Bibliya at kung gaano kapraktikal ang Bibliya sa ngayon",
      "__label__tl",
      0.85911,
   );
}

#[test]
fn golden_ftz_tr() {
   let model = load_model();
   assert_top1(
      &model,
      "Kutsal Kitap 66 kitaptan olu\u{015f}ur ve yakla\u{015f}\u{0131}k 1.600 y\u{0131}ll\u{0131}k bir s\u{00fc}rede yaz\u{0131}lm\u{0131}\u{015f}t\u{0131}r",
      "__label__tr",
      0.997454,
   );
}

#[test]
fn golden_ftz_sw() {
   let model = load_model();
   assert_top1(
      &model,
      "Tafuta majibu ya maswali ya Biblia na upate msaada kwa ajili ya familia yako",
      "__label__sw",
      0.775359,
   );
}

#[test]
fn golden_ftz_ceb() {
   let model = load_model();
   assert_top1(
      &model,
      "Gipakita sa mga topiko niini nga seksiyon kon nganong makasalig ka sa Bibliya ug kon unsa ka praktikal ang Bibliya",
      "__label__ceb",
      0.909811,
   );
}

// --- Golden parity: CJK and Vietnamese ---

#[test]
fn golden_ftz_ja() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{3053}\u{306e}\u{30bb}\u{30af}\u{30b7}\u{30e7}\u{30f3}\u{306f}\u{ff0c}\u{8056}\u{66f8}\u{3092}\u{4fe1}\u{983c}\u{3067}\u{304d}\u{308b}\u{306e}\u{306f}\u{306a}\u{305c}\u{304b}\u{ff0c}\u{8056}\u{66f8}\u{3092}\u{6700}\u{5927}\u{9650}\u{306b}\u{6d3b}\u{7528}\u{3059}\u{308b}\u{306b}\u{306f}\u{3069}\u{3046}\u{3059}\u{308c}\u{3070}\u{3088}\u{3044}\u{304b}\u{ff0c}\u{8056}\u{66f8}\u{304c}\u{3069}\u{308c}\u{307b}\u{3069}\u{5f79}\u{7acb}\u{3064}\u{304b}\u{3092}\u{53d6}\u{308a}\u{4e0a}\u{3052}\u{3066}\u{3044}\u{307e}\u{3059}",
      "__label__ja",
      0.999871,
   );
}

#[test]
fn golden_ftz_zh() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{5723}\u{7ecf}\u{503c}\u{5f97}\u{4e00}\u{770b}\u{5417}\u{ff1f}\u{5723}\u{7ecf}\u{5bf9}\u{6211}\u{6709}\u{5e2e}\u{52a9}\u{5417}\u{ff1f}\u{8bf7}\u{770b}\u{770b}\u{8fd9}\u{4e2a}\u{4e13}\u{680f}\u{7684}\u{8d44}\u{6599}\u{ff0c}\u{4f60}\u{4f1a}\u{53d1}\u{73b0}\u{9605}\u{8bfb}\u{5723}\u{7ecf}\u{7684}\u{786e}\u{5927}\u{6709}\u{4ef7}\u{503c}",
      "__label__zh",
      0.999156,
   );
}

#[test]
fn golden_ftz_ko() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{c774} \u{c139}\u{c158}\u{c5d0} \u{b098}\u{c624}\u{b294} \u{c8fc}\u{c81c}\u{b4e4}\u{c744} \u{c0b4}\u{d3b4}\u{bcf4}\u{ba74} \u{c131}\u{acbd}\u{c744} \u{c2e0}\u{b8b0}\u{d560} \u{c218} \u{c788}\u{b294} \u{c774}\u{c720}\u{ac00} \u{bb34}\u{c5c7}\u{c778}\u{c9c0} \u{c5b4}\u{b5bb}\u{ac8c} \u{c131}\u{acbd}\u{c5d0}\u{c11c} \u{cd5c}\u{b300}\u{c758} \u{c720}\u{c775}\u{c744} \u{c5bb}\u{c744} \u{c218} \u{c788}\u{b294}\u{c9c0} \u{c54c} \u{c218} \u{c788}\u{c2b5}\u{b2c8}\u{b2e4}",
      "__label__ko",
      1.00007,
   );
}

#[test]
fn golden_ftz_vi() {
   let model = load_model();
   assert_top1(
      &model,
      "Nh\u{1eef}ng ch\u{1ee7} \u{0111}\u{1ec1} trong ph\u{1ea7}n n\u{00e0}y cho th\u{1ea5}y l\u{00fd} do c\u{00f3} th\u{1ec3} tin c\u{1ead}y Kinh Th\u{00e1}nh v\u{00e0} Kinh Th\u{00e1}nh th\u{1ead}t s\u{1ef1} thi\u{1ebf}t th\u{1ef1}c nh\u{01b0} th\u{1ebf} n\u{00e0}o",
      "__label__vi",
      0.997548,
   );
}

// --- Golden parity: Cyrillic ---

#[test]
fn golden_ftz_ru() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{0411}\u{0438}\u{0431}\u{043b}\u{0438}\u{044f} \u{0441}\u{043e}\u{0441}\u{0442}\u{043e}\u{0438}\u{0442} \u{0438}\u{0437} 66 \u{043a}\u{043d}\u{0438}\u{0433}. \u{0418}\u{0445} \u{043d}\u{0430}\u{043f}\u{0438}\u{0441}\u{0430}\u{043d}\u{0438}\u{0435} \u{0437}\u{0430}\u{043d}\u{044f}\u{043b}\u{043e} \u{043e}\u{043a}\u{043e}\u{043b}\u{043e} 1 600 \u{043b}\u{0435}\u{0442}. \u{0412} \u{0411}\u{0438}\u{0431}\u{043b}\u{0438}\u{0438} \u{0441}\u{043e}\u{0434}\u{0435}\u{0440}\u{0436}\u{0438}\u{0442}\u{0441}\u{044f} \u{0441}\u{043e}\u{043e}\u{0431}\u{0449}\u{0435}\u{043d}\u{0438}\u{0435} \u{043e}\u{0442} \u{0411}\u{043e}\u{0433}\u{0430}",
      "__label__ru",
      0.990873,
   );
}

#[test]
fn golden_ftz_uk() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{0411}\u{0456}\u{0431}\u{043b}\u{0456}\u{044f} \u{2014} \u{0446}\u{0435} \u{0437}\u{0431}\u{0456}\u{0440}\u{043a}\u{0430} 66 \u{0441}\u{0432}\u{044f}\u{0449}\u{0435}\u{043d}\u{043d}\u{0438}\u{0445} \u{043a}\u{043d}\u{0438}\u{0433}, \u{044f}\u{043a}\u{0430} \u{043f}\u{0438}\u{0441}\u{0430}\u{043b}\u{0430}\u{0441}\u{044f} \u{0432}\u{043f}\u{0440}\u{043e}\u{0434}\u{043e}\u{0432}\u{0436} 1600 \u{0440}\u{043e}\u{043a}\u{0456}\u{0432} \u{0456} \u{043c}\u{0456}\u{0441}\u{0442}\u{0438}\u{0442}\u{044c} \u{0411}\u{043e}\u{0436}\u{0456} \u{0441}\u{043b}\u{043e}\u{0432}\u{0430}",
      "__label__uk",
      0.996879,
   );
}

// --- Golden parity: Arabic script ---

#[test]
fn golden_ftz_ar() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{0627}\u{0644}\u{0643}\u{062a}\u{0627}\u{0628} \u{0627}\u{0644}\u{0645}\u{0642}\u{062f}\u{0633} \u{0647}\u{0648} \u{0645}\u{062c}\u{0645}\u{0648}\u{0639}\u{0629} \u{0645}\u{0646} \u{0666}\u{0666} \u{0633}\u{0641}\u{0631}\u{064b}\u{0627} \u{0643}\u{064f}\u{062a}\u{0628}\u{062a} \u{0639}\u{0644}\u{0649} \u{0645}\u{062f}\u{0649} \u{0666}\u{0660}\u{0660} \u{0633}\u{0646}\u{0629} \u{062a}\u{0642}\u{0631}\u{064a}\u{0628}\u{064b}\u{0627} \u{0648}\u{064a}\u{062d}\u{062a}\u{0648}\u{064a} \u{0639}\u{0644}\u{0649} \u{0631}\u{0633}\u{0627}\u{0644}\u{0629} \u{0627}\u{0644}\u{0644}\u{0647} \u{0644}\u{0644}\u{0628}\u{0634}\u{0631}",
      "__label__ar",
      0.967601,
   );
}

#[test]
fn golden_ftz_ur() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{0627}\u{0650}\u{0633} \u{062d}\u{0635}\u{06d2} \u{0645}\u{06cc}\u{06ba} \u{0622}\u{067e} \u{062f}\u{06cc}\u{06a9}\u{06be} \u{067e}\u{0627}\u{0626}\u{06cc}\u{06ba} \u{06af}\u{06d2} \u{06a9}\u{06c1} \u{0622}\u{067e} \u{0628}\u{0627}\u{0626}\u{0628}\u{0644} \u{067e}\u{0631} \u{0628}\u{06be}\u{0631}\u{0648}\u{0633}\u{0627} \u{06a9}\u{06cc}\u{0648}\u{06ba} \u{06a9}\u{0631} \u{0633}\u{06a9}\u{062a}\u{06d2} \u{06c1}\u{06cc}\u{06ba} \u{0627}\u{0648}\u{0631} \u{0627}\u{0650}\u{0633} \u{0645}\u{06cc}\u{06ba} \u{062f}\u{0631}\u{062c} \u{0645}\u{0634}\u{0648}\u{0631}\u{06d2} \u{06a9}\u{062a}\u{0646}\u{06d2} \u{0639}\u{0645}\u{0644}\u{06cc} \u{06c1}\u{06cc}\u{06ba}",
      "__label__ur",
      0.979621,
   );
}

#[test]
fn golden_ftz_fa() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{0645}\u{0648}\u{0636}\u{0648}\u{0639}\u{0627}\u{062a} \u{0627}\u{06cc}\u{0646} \u{0628}\u{062e}\u{0634} \u{0646}\u{0634}\u{0627}\u{0646} \u{0645}\u{06cc}\u{200c}\u{062f}\u{0647}\u{062f} \u{06a9}\u{0647} \u{0686}\u{0631}\u{0627} \u{0645}\u{06cc}\u{200c}\u{062a}\u{0648}\u{0627}\u{0646} \u{0628}\u{0647} \u{06a9}\u{062a}\u{0627}\u{0628} \u{0645}\u{0642}\u{062f}\u{0651}\u{0633} \u{0627}\u{0639}\u{062a}\u{0645}\u{0627}\u{062f} \u{06a9}\u{0631}\u{062f} \u{0648} \u{06a9}\u{062a}\u{0627}\u{0628} \u{0645}\u{0642}\u{062f}\u{0651}\u{0633} \u{062a}\u{0627} \u{0686}\u{0647} \u{062d}\u{062f} \u{06a9}\u{0627}\u{0631}\u{0622}\u{06cc}\u{06cc} \u{062f}\u{0627}\u{0631}\u{062f}",
      "__label__fa",
      0.955136,
   );
}

// --- Golden parity: Indic ---

#[test]
fn golden_ftz_hi() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{0907}\u{0938} \u{092d}\u{093e}\u{0917} \u{092e}\u{0947}\u{0902} \u{0906}\u{092a} \u{0926}\u{0947}\u{0916}\u{0947}\u{0902}\u{0917}\u{0947} \u{0915}\u{093f} \u{0906}\u{092a} \u{092a}\u{0935}\u{093f}\u{0924}\u{094d}\u{0930} \u{0936}\u{093e}\u{0938}\u{094d}\u{0924}\u{094d}\u{0930} \u{092a}\u{0930} \u{0915}\u{094d}\u{092f}\u{094b}\u{0902} \u{092f}\u{0915}\u{0940}\u{0928} \u{0915}\u{0930} \u{0938}\u{0915}\u{0924}\u{0947} \u{0939}\u{0948}\u{0902} \u{0914}\u{0930} \u{092a}\u{0935}\u{093f}\u{0924}\u{094d}\u{0930} \u{0936}\u{093e}\u{0938}\u{094d}\u{0924}\u{094d}\u{0930} \u{092e}\u{0947}\u{0902} \u{0926}\u{0940} \u{0938}\u{0932}\u{093e}\u{0939} \u{0906}\u{091c} \u{0939}\u{092e}\u{093e}\u{0930}\u{0947} \u{0932}\u{093f}\u{090f} \u{0915}\u{093f}\u{0924}\u{0928}\u{0940} \u{092b}\u{093e}\u{092f}\u{0926}\u{0947}\u{092e}\u{0902}\u{0926} \u{0939}\u{0948}",
      "__label__hi",
      0.988689,
   );
}

#[test]
fn golden_ftz_bn() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{098f}\u{0987} \u{09ac}\u{09bf}\u{09ad}\u{09be}\u{0997}\u{09c7}\u{09b0} \u{09ac}\u{09bf}\u{09b7}\u{09af}\u{09bc}\u{0997}\u{09c1}\u{09b2}\u{09cb} \u{09a6}\u{09c7}\u{0996}\u{09be}\u{09af}\u{09bc} \u{09af}\u{09c7} \u{0995}\u{09c7}\u{09a8} \u{0986}\u{09aa}\u{09a8}\u{09bf} \u{09ac}\u{09be}\u{0987}\u{09ac}\u{09c7}\u{09b2}\u{09c7}\u{09b0} \u{0989}\u{09aa}\u{09b0} \u{0986}\u{09b8}\u{09cd}\u{09a5}\u{09be} \u{09b0}\u{09be}\u{0996}\u{09a4}\u{09c7} \u{09aa}\u{09be}\u{09b0}\u{09c7}\u{09a8} \u{098f}\u{09ac}\u{0982} \u{09ac}\u{09be}\u{0987}\u{09ac}\u{09c7}\u{09b2} \u{09aa}\u{09cd}\u{09b0}\u{0995}\u{09c3}\u{09a4}\u{09aa}\u{0995}\u{09cd}\u{09b7}\u{09c7} \u{0995}\u{09a4}\u{099f}\u{09be} \u{09ac}\u{09cd}\u{09af}\u{09be}\u{09ac}\u{09b9}\u{09be}\u{09b0}\u{09bf}\u{0995}",
      "__label__bn",
      0.990291,
   );
}

#[test]
fn golden_ftz_ta() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{0baa}\u{0bc8}\u{0baa}\u{0bbf}\u{0bb3}\u{0bc1}\u{0b95}\u{0bcd}\u{0b95}\u{0bc1} \u{0b95}\u{0b9f}\u{0bb5}\u{0bc1}\u{0bb3}\u{0bc1}\u{0b9f}\u{0bc8}\u{0baf} \u{0bb5}\u{0bbe}\u{0bb0}\u{0bcd}\u{0ba4}\u{0bcd}\u{0ba4}\u{0bc8} \u{0b8e}\u{0ba9}\u{0bcd}\u{0bb1} \u{0baa}\u{0bc6}\u{0baf}\u{0bb0}\u{0bcd} \u{0b87}\u{0bb0}\u{0bc1}\u{0b95}\u{0bcd}\u{0b95}\u{0bbf}\u{0bb1}\u{0ba4}\u{0bc1} \u{0b8f}\u{0ba9}\u{0bc6}\u{0ba9}\u{0bcd}\u{0bb1}\u{0bbe}\u{0bb2}\u{0bcd} \u{0b95}\u{0b9f}\u{0bb5}\u{0bc1}\u{0bb3}\u{0bcd} \u{0bae}\u{0b95}\u{0bcd}\u{0b95}\u{0bb3}\u{0bc1}\u{0b95}\u{0bcd}\u{0b95}\u{0bc1} \u{0b9a}\u{0bca}\u{0ba9}\u{0bcd}\u{0ba9} \u{0b9a}\u{0bc6}\u{0baf}\u{0bcd}\u{0ba4}\u{0bbf}\u{0ba4}\u{0bbe}\u{0ba9}\u{0bcd} \u{0b85}\u{0ba4}\u{0bbf}\u{0bb2}\u{0bcd} \u{0b87}\u{0bb0}\u{0bc1}\u{0b95}\u{0bcd}\u{0b95}\u{0bbf}\u{0bb1}\u{0ba4}\u{0bc1}",
      "__label__ta",
      0.99989,
   );
}

// --- Golden parity: Southeast Asian ---

#[test]
fn golden_ftz_th() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{0e2b}\u{0e31}\u{0e27}\u{0e02}\u{0e49}\u{0e2d}\u{0e15}\u{0e48}\u{0e32}\u{0e07} \u{0e46} \u{0e43}\u{0e19}\u{0e2a}\u{0e48}\u{0e27}\u{0e19}\u{0e19}\u{0e35}\u{0e49}\u{0e0a}\u{0e48}\u{0e27}\u{0e22}\u{0e43}\u{0e2b}\u{0e49}\u{0e04}\u{0e38}\u{0e13}\u{0e40}\u{0e2b}\u{0e47}\u{0e19}\u{0e27}\u{0e48}\u{0e32}\u{0e17}\u{0e33}\u{0e44}\u{0e21}\u{0e04}\u{0e38}\u{0e13}\u{0e40}\u{0e0a}\u{0e37}\u{0e48}\u{0e2d}\u{0e2a}\u{0e34}\u{0e48}\u{0e07}\u{0e17}\u{0e35}\u{0e48}\u{0e04}\u{0e31}\u{0e21}\u{0e20}\u{0e35}\u{0e23}\u{0e4c}\u{0e44}\u{0e1a}\u{0e40}\u{0e1a}\u{0e34}\u{0e25}\u{0e1a}\u{0e2d}\u{0e01}\u{0e44}\u{0e14}\u{0e49} \u{0e41}\u{0e25}\u{0e30}\u{0e04}\u{0e31}\u{0e21}\u{0e20}\u{0e35}\u{0e23}\u{0e4c}\u{0e44}\u{0e1a}\u{0e40}\u{0e1a}\u{0e34}\u{0e25}\u{0e21}\u{0e35}\u{0e04}\u{0e33}\u{0e41}\u{0e19}\u{0e30}\u{0e19}\u{0e33}\u{0e17}\u{0e35}\u{0e48}\u{0e14}\u{0e35}\u{0e02}\u{0e19}\u{0e32}\u{0e14}\u{0e44}\u{0e2b}\u{0e19}",
      "__label__th",
      0.999499,
   );
}

#[test]
fn golden_ftz_my() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{1000}\u{103b}\u{1019}\u{103a}\u{1038}\u{1005}\u{102c}\u{1019}\u{1031}\u{1038}\u{1001}\u{103d}\u{1014}\u{103a}\u{1038}\u{1010}\u{103d}\u{1031}\u{101b}\u{1032}\u{1037} \u{1021}\u{1016}\u{103c}\u{1031}\u{1000}\u{102d}\u{102f}\u{1000}\u{103c}\u{100a}\u{1037}\u{103a}\u{1015}\u{102b} \u{101e}\u{1004}\u{1037}\u{103a}\u{1019}\u{102d}\u{101e}\u{102c}\u{1038}\u{1005}\u{102f}\u{1021}\u{1010}\u{103d}\u{1000}\u{103a} \u{101c}\u{1000}\u{103a}\u{1010}\u{103d}\u{1031}\u{1037}\u{1000}\u{103b}\u{1010}\u{1032}\u{1037} \u{1021}\u{1000}\u{1030}\u{1021}\u{100a}\u{102e}\u{1000}\u{102d}\u{102f}\u{101a}\u{1030}\u{1015}\u{102b}",
      "__label__my",
      0.999314,
   );
}

#[test]
fn golden_ftz_km() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{1794}\u{17d2}\u{179a}\u{1792}\u{17b6}\u{1793}\u{1794}\u{1791}\u{1795}\u{17d2}\u{179f}\u{17c1}\u{1784}\u{17d7}\u{1780}\u{17d2}\u{1793}\u{17bb}\u{1784}\u{1795}\u{17d2}\u{1793}\u{17c2}\u{1780}\u{1793}\u{17c1}\u{17c7}\u{17a2}\u{17b6}\u{1785}\u{1787}\u{17bd}\u{1799}\u{179b}\u{17c4}\u{1780}\u{17a2}\u{17d2}\u{1793}\u{1780}\u{179f}\u{17d2}\u{179c}\u{17c2}\u{1784}\u{1799}\u{179b}\u{17cb}\u{17a2}\u{17c6}\u{1796}\u{17b8}\u{1798}\u{17bc}\u{179b}\u{17a0}\u{17c1}\u{178f}\u{17bb}\u{178a}\u{17c2}\u{179b}\u{179b}\u{17c4}\u{1780}\u{17a2}\u{17d2}\u{1793}\u{1780}\u{17a2}\u{17b6}\u{1785}\u{1791}\u{17bb}\u{1780}\u{1785}\u{17b7}\u{178f}\u{17d2}\u{178f}\u{1780}\u{17b6}\u{179a}\u{178e}\u{17c2}\u{1793}\u{17b6}\u{17c6}\u{1780}\u{17d2}\u{1793}\u{17bb}\u{1784}\u{1782}\u{1798}\u{17d2}\u{1796}\u{17b8}\u{179a}",
      "__label__km",
      0.997727,
   );
}

// --- Golden parity: Other scripts ---

#[test]
fn golden_ftz_el() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{03a3}\u{03b5} \u{03b1}\u{03c5}\u{03c4}\u{03cc} \u{03c4}\u{03bf} \u{03c4}\u{03bc}\u{03ae}\u{03bc}\u{03b1} \u{03b8}\u{03b1} \u{03bc}\u{03ac}\u{03b8}\u{03b5}\u{03c4}\u{03b5} \u{03b3}\u{03b9}\u{03b1}\u{03c4}\u{03af} \u{03bc}\u{03c0}\u{03bf}\u{03c1}\u{03b5}\u{03af}\u{03c4}\u{03b5} \u{03bd}\u{03b1} \u{03b5}\u{03bc}\u{03c0}\u{03b9}\u{03c3}\u{03c4}\u{03b5}\u{03c5}\u{03c4}\u{03b5}\u{03af}\u{03c4}\u{03b5} \u{03c4}\u{03b7}\u{03bd} \u{0391}\u{03b3}\u{03af}\u{03b1} \u{0393}\u{03c1}\u{03b1}\u{03c6}\u{03ae} \u{03ba}\u{03b1}\u{03b9} \u{03c0}\u{03cc}\u{03c3}\u{03bf} \u{03c0}\u{03c1}\u{03b1}\u{03ba}\u{03c4}\u{03b9}\u{03ba}\u{03ae} \u{03b5}\u{03af}\u{03bd}\u{03b1}\u{03b9}",
      "__label__el",
      0.998962,
   );
}

#[test]
fn golden_ftz_he() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{05d4}\u{05e0}\u{05d5}\u{05e9}\u{05d0}\u{05d9}\u{05dd} \u{05d4}\u{05e0}\u{05d9}\u{05d3}\u{05d5}\u{05e0}\u{05d9}\u{05dd} \u{05d1}\u{05de}\u{05d3}\u{05d5}\u{05e8} \u{05d6}\u{05d4} \u{05de}\u{05d1}\u{05dc}\u{05d9}\u{05d8}\u{05d9}\u{05dd} \u{05de}\u{05d3}\u{05d5}\u{05e2} \u{05d0}\u{05ea}\u{05d4} \u{05d9}\u{05db}\u{05d5}\u{05dc} \u{05dc}\u{05d1}\u{05d8}\u{05d5}\u{05d7} \u{05d1}\u{05de}\u{05e7}\u{05e8}\u{05d0} \u{05d5}\u{05db}\u{05d9}\u{05e6}\u{05d3} \u{05d0}\u{05ea}\u{05d4} \u{05d9}\u{05db}\u{05d5}\u{05dc} \u{05dc}\u{05d4}\u{05e4}\u{05d9}\u{05e7} \u{05de}\u{05de}\u{05e0}\u{05d5} \u{05ea}\u{05d5}\u{05e2}\u{05dc}\u{05ea} \u{05e8}\u{05d1}\u{05d4} \u{05d9}\u{05d5}\u{05ea}\u{05e8}",
      "__label__he",
      0.999913,
   );
}

#[test]
fn golden_ftz_am() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{1218}\u{133d}\u{1210}\u{134d} \u{1245}\u{12f1}\u{1235} \u{12e8} 66 \u{1245}\u{12f1}\u{1235} \u{1218}\u{133b}\u{1215}\u{134d}\u{1275} \u{1235}\u{1265}\u{1235}\u{1265} \u{1290}\u{12cd}\u{1362} \u{12e8}\u{1270}\u{133b}\u{1348}\u{12cd}\u{121d} \u{1260} 1,600 \u{12d3}\u{1218}\u{1273}\u{1275} \u{12cd}\u{1235}\u{1325} \u{1290}\u{12cd}",
      "__label__am",
      0.999573,
   );
}

#[test]
fn golden_ftz_ka() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{10d0}\u{10db} \u{10d2}\u{10d0}\u{10dc}\u{10e7}\u{10dd}\u{10e4}\u{10d8}\u{10da}\u{10d4}\u{10d1}\u{10d0}\u{10e8}\u{10d8} \u{10d2}\u{10d0}\u{10dc}\u{10d7}\u{10d0}\u{10d5}\u{10e1}\u{10d4}\u{10d1}\u{10e3}\u{10da}\u{10d8} \u{10e1}\u{10e2}\u{10d0}\u{10e2}\u{10d8}\u{10d4}\u{10d1}\u{10d8}\u{10d3}\u{10d0}\u{10dc} \u{10d2}\u{10d0}\u{10d8}\u{10d2}\u{10d4}\u{10d1}\u{10d7} \u{10e0}\u{10d0}\u{10e2}\u{10dd}\u{10db} \u{10e8}\u{10d4}\u{10d2}\u{10d8}\u{10eb}\u{10da}\u{10d8}\u{10d0}\u{10d7} \u{10d4}\u{10dc}\u{10d3}\u{10dd}\u{10d7} \u{10d1}\u{10d8}\u{10d1}\u{10da}\u{10d8}\u{10d0}\u{10e1} \u{10d3}\u{10d0} \u{10e0}\u{10d0}\u{10db}\u{10d3}\u{10d4}\u{10dc}\u{10d0}\u{10d3} \u{10de}\u{10e0}\u{10d0}\u{10e5}\u{10e2}\u{10d8}\u{10d9}\u{10e3}\u{10da}\u{10d8}\u{10d0} \u{10db}\u{10d0}\u{10e1}\u{10e8}\u{10d8} \u{10e9}\u{10d0}\u{10ec}\u{10d4}\u{10e0}\u{10d8}\u{10da}\u{10d8} \u{10e0}\u{10e9}\u{10d4}\u{10d5}\u{10d4}\u{10d1}\u{10d8}",
      "__label__ka",
      0.996873,
   );
}

#[test]
fn golden_ftz_hy() {
   let model = load_model();
   assert_top1(
      &model,
      "\u{0533}\u{057f}\u{0565}\u{0584} \u{0561}\u{057d}\u{057f}\u{057e}\u{0561}\u{056e}\u{0561}\u{0577}\u{0576}\u{0579}\u{0575}\u{0561}\u{0576} \u{0570}\u{0561}\u{0580}\u{0581}\u{0565}\u{0580}\u{056b} \u{057a}\u{0561}\u{057f}\u{0561}\u{057d}\u{056d}\u{0561}\u{0576}\u{0576}\u{0565}\u{0580}\u{0568} \u{0587} \u{0563}\u{0578}\u{0580}\u{056e}\u{0576}\u{0561}\u{056f}\u{0561}\u{0576} \u{056d}\u{0578}\u{0580}\u{0570}\u{0578}\u{0582}\u{0580}\u{0564}\u{0576}\u{0565}\u{0580} \u{057d}\u{057f}\u{0561}\u{0581}\u{0565}\u{0584} \u{0568}\u{0576}\u{057f}\u{0561}\u{0576}\u{056b}\u{0584}\u{056b} \u{057e}\u{0565}\u{0580}\u{0561}\u{0562}\u{0565}\u{0580}\u{0575}\u{0561}\u{056c}",
      "__label__hy",
      0.999938,
   );
}

// --- Structural tests ---

#[test]
fn predictions_ftz_sorted_descending() {
   let model = load_model();
   let predictions = model.predict("hello world", 5, 0.0).unwrap();
   for i in 1..predictions.len() {
      assert!(
         predictions[i - 1].probability >= predictions[i].probability,
         "predictions not sorted: {} < {}",
         predictions[i - 1].probability,
         predictions[i].probability,
      );
   }
}

#[test]
fn threshold_ftz_filters_results() {
   let model = load_model();
   let predictions = model
      .predict(
         "The Bible is a collection of 66 sacred books written over a period of approximately 1,600 years",
         10,
         0.99,
      )
      .unwrap();
   for p in &predictions {
      assert!(
         p.probability >= 0.99,
         "probability {} < 0.99",
         p.probability
      );
   }
}

#[test]
fn golden_ftz_sentence_vector() {
   let model = load_model();
   let vector = model.get_sentence_vector("hello world").unwrap();
   assert_eq!(vector.len(), 16);

   // Golden values from C++ `print-sentence-vectors`
   let expected: [f32; 16] = [
      -0.65101, 0.10776, -0.35615, -0.18231, 0.084701, 0.17492, -0.22578, 0.1906, 0.064585,
      -0.28229, 0.076955, 0.0064092, 0.062685, 0.16378, 0.12939, 0.039792,
   ];

   for (i, (&actual, &exp)) in vector.iter().zip(expected.iter()).enumerate() {
      let diff = (actual - exp).abs();
      assert!(
         diff < 1e-4,
         "sentence vector dim {i}: expected {exp}, got {actual} (diff {diff})",
      );
   }
}

#[test]
fn empty_text_ftz_returns_ok() {
   // Empty input still produces tokens because get_line always appends the
   // EOS marker ("</s>"), so predict succeeds rather than returning
   // EmptyInput.
   let model = load_model();
   let result = model.predict("", 1, 0.0);
   assert!(
      result.is_ok(),
      "expected Ok for empty input, got {result:?}"
   );
}

#[test]
fn whitespace_only_ftz_returns_ok() {
   // Whitespace-only input behaves like empty input: split_whitespace
   // yields nothing, but the EOS marker still produces tokens.
   let model = load_model();
   let result = model.predict("   ", 1, 0.0);
   assert!(
      result.is_ok(),
      "expected Ok for whitespace-only input, got {result:?}"
   );
}

#[test]
fn k_zero_returns_empty_vec() {
   let model = load_model();
   let preds = model.predict("This is English", 0, 0.0).unwrap();
   assert!(
      preds.is_empty(),
      "expected empty vec for k=0, got {} results",
      preds.len()
   );
}

#[test]
fn k_exceeding_label_count_returns_at_most_176() {
   let model = load_model();
   let preds = model.predict("This is English", 500, 0.0).unwrap();
   assert!(
      preds.len() <= 176,
      "expected at most 176 results, got {}",
      preds.len(),
   );
}

#[test]
fn long_single_token_exercises_heap_fallback() {
   // A single token longer than 258 bytes (including markers) exercises
   // the heap-allocation fallback in compute_sub_words_with_markers.
   let model = load_model();
   let long_token = "a".repeat(300);
   let result = model.predict(&long_token, 1, 0.0);
   assert!(
      result.is_ok(),
      "expected Ok for long single token, got {result:?}"
   );
}
