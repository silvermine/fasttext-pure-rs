//! Tier 3: Benchmark dataset validation tests.
//!
//! These tests run the lid.176.ftz model against known sentences and
//! verify aggregate accuracy. They are marked `#[ignore]` because they
//! take significant time or require external datasets.
//!
//! Run with: `cargo test -- --ignored`

use fasttext_pure_rs::FastText;

const MODEL_PATH: &str = "tests/fixtures/lid.176.ftz";

/// A representative set of sentences covering many of the 176 supported
/// languages, with known expected labels.
const BENCHMARK_SENTENCES: &[(&str, &str)] = &[
   // European languages
   ("This is a simple test in English", "__label__en"),
   ("I went to the store to buy some milk", "__label__en"),
   ("The weather today is absolutely beautiful", "__label__en"),
   ("She writes code in Python and Rust", "__label__en"),
   ("Ceci est un test simple en francais", "__label__fr"),
   ("Je suis alle au magasin acheter du lait", "__label__fr"),
   ("Le temps est magnifique aujourdhui", "__label__fr"),
   ("Dies ist ein einfacher Test auf Deutsch", "__label__de"),
   (
      "Ich bin zum Laden gegangen um Milch zu kaufen",
      "__label__de",
   ),
   ("Das Wetter heute ist wirklich wunderschoen", "__label__de"),
   ("Esta es una prueba sencilla en espanol", "__label__es"),
   ("Fui a la tienda a comprar un poco de leche", "__label__es"),
   ("El tiempo hoy esta absolutamente hermoso", "__label__es"),
   ("Questa e una semplice prova in italiano", "__label__it"),
   ("Sono andato al negozio a comprare del latte", "__label__it"),
   (
      "Dit is een eenvoudige test in het Nederlands",
      "__label__nl",
   ),
   ("Ik ging naar de winkel om melk te kopen", "__label__nl"),
   ("Isto e um teste simples em portugues", "__label__pt"),
   ("Fui para a loja comprar um pouco de leite", "__label__pt"),
   (
      "To jest prosty test po polsku i chcę to sprawdzić",
      "__label__pl",
   ),
   ("Poszedłem do sklepu po mleko i chleb", "__label__pl"),
   ("Det har ar ett enkelt test pa svenska", "__label__sv"),
   ("Jag gick till butiken for att kopa mjolk", "__label__sv"),
   ("Tama on yksinkertainen testi suomeksi", "__label__fi"),
   ("Menin kauppaan ostamaan maitoa ja leipää", "__label__fi"),
   (
      "Toto je jednoduchý test v češtině a funguje to",
      "__label__cs",
   ),
   ("Šel jsem do obchodu koupit mléko", "__label__cs"),
   ("Aceasta este un test simplu in limba romana", "__label__ro"),
   ("Am mers la magazin să cumpăr lapte și pâine", "__label__ro"),
   ("Ez egy egyszerű teszt magyarul írva", "__label__hu"),
   ("Elmentem a boltba tejet és kenyeret venni", "__label__hu"),
   ("Bu bir Turkce test cumlesidir", "__label__tr"),
   ("Dukkana sut almaya gittim bugun", "__label__tr"),
   ("Dette er en enkel test paa dansk", "__label__da"),
   ("Jeg gikk til butikken for a kjope melk", "__label__no"),
   ("Αυτή είναι μια απλή δοκιμή στα ελληνικά", "__label__el"),
   ("Πήγα στο μαγαζί να αγοράσω γάλα", "__label__el"),
   // CJK languages
   ("これは日本語の簡単なテストです", "__label__ja"),
   ("牛乳を買いに店に行きました", "__label__ja"),
   ("今日の天気は本当に素晴らしいです", "__label__ja"),
   ("这是一个简单的中文测试", "__label__zh"),
   ("我去商店买了一些牛奶", "__label__zh"),
   ("今天的天气真是太好了", "__label__zh"),
   ("이것은 한국어로 된 간단한 테스트입니다", "__label__ko"),
   ("우유를 사러 가게에 갔습니다", "__label__ko"),
   ("오늘 날씨가 정말 좋습니다", "__label__ko"),
   // Other scripts
   ("Это простой тест на русском языке", "__label__ru"),
   ("Я пошел в магазин купить молока", "__label__ru"),
   ("Сегодня погода просто прекрасная", "__label__ru"),
   ("Це простий тест українською мовою", "__label__uk"),
   ("Я пішов до магазину купити молока", "__label__uk"),
   ("هذا اختبار بسيط باللغة العربية", "__label__ar"),
   ("ذهبت إلى المتجر لشراء الحليب", "__label__ar"),
   ("นี่คือการทดสอบง่ายๆในภาษาไทย", "__label__th"),
   ("ฉันไปที่ร้านเพื่อซื้อนม", "__label__th"),
   (
      "Đây là một bài kiểm tra đơn giản bằng tiếng Việt",
      "__label__vi",
   ),
   ("Tôi đã đi đến cửa hàng để mua sữa", "__label__vi"),
   ("यह हिंदी में एक सरल परीक्षा है", "__label__hi"),
   ("मैं दूध खरीदने दुकान गया", "__label__hi"),
   ("এটি বাংলায় একটি সহজ পরীক্ষা", "__label__bn"),
   // Southeast Asian & others
   (
      "Ini adalah ujian mudah dalam bahasa Melayu dan sangat senang",
      "__label__ms",
   ),
   (
      "Ini adalah tes sederhana dalam bahasa Indonesia",
      "__label__id",
   ),
   ("Ito ay isang simpleng pagsubok sa Filipino", "__label__tl"),
];

/// Run all benchmark sentences and compute precision@1.
///
/// This test is ignored by default because it validates aggregate
/// accuracy rather than individual correctness.
#[test]
#[ignore]
fn benchmark_precision_at_1() {
   let model = FastText::load(MODEL_PATH).expect("failed to load model");
   let total = BENCHMARK_SENTENCES.len();
   let mut correct = 0;
   let mut failures: Vec<(&str, &str, String)> = Vec::new();

   for &(text, expected_label) in BENCHMARK_SENTENCES {
      let predictions = model.predict(text, 1, 0.0).unwrap();
      if predictions[0].label == expected_label {
         correct += 1;
      } else {
         failures.push((text, expected_label, predictions[0].label.clone()));
      }
   }

   let precision = correct as f64 / total as f64;
   eprintln!("Benchmark: {correct}/{total} correct (precision@1 = {precision:.4})");

   if !failures.is_empty() {
      eprintln!("\nFailures:");
      for (text, expected, actual) in &failures {
         eprintln!("  expected={expected} actual={actual} text=\"{text}\"");
      }
   }

   // We expect very high accuracy for clear, unambiguous sentences
   assert!(
      precision >= 0.95,
      "precision@1 = {precision:.4} is below threshold 0.95 ({correct}/{total})",
   );
}

/// Verify that model loading time is reasonable (under 1 second).
#[test]
#[ignore]
fn benchmark_load_time() {
   let start = std::time::Instant::now();
   let _model = FastText::load(MODEL_PATH).expect("failed to load model");
   let elapsed = start.elapsed();
   eprintln!("Model load time: {elapsed:?}");
   assert!(
      elapsed.as_secs() < 1,
      "model loading took {elapsed:?}, expected < 1s",
   );
}

/// Verify that prediction throughput is reasonable (> 1000 predictions/sec).
#[test]
#[ignore]
fn benchmark_prediction_throughput() {
   let model = FastText::load(MODEL_PATH).expect("failed to load model");
   let sentences = [
      "This is a test sentence in English",
      "Ceci est une phrase en francais",
      "Dies ist ein Satz auf Deutsch",
      "Esta es una oracion en espanol",
      "これは日本語のテストです",
   ];

   let iterations = 10_000;
   let start = std::time::Instant::now();
   for i in 0..iterations {
      let text = sentences[i % sentences.len()];
      let _ = model.predict(text, 1, 0.0).unwrap();
   }
   let elapsed = start.elapsed();
   let throughput = iterations as f64 / elapsed.as_secs_f64();
   eprintln!(
      "Prediction throughput: {throughput:.0} predictions/sec ({iterations} in {elapsed:?})"
   );
   assert!(
      throughput > 10_000.0,
      "throughput {throughput:.0}/sec is below 10,000/sec",
   );
}
