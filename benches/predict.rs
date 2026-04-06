//! Benchmarks for the fasttext-pure-rs prediction pipeline.
//!
//! Run with: `cargo bench`
//!
//! These benchmarks measure end-to-end prediction latency across
//! different input lengths, top-k values, and model loading time.
//! Results include statistical analysis with confidence intervals.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fasttext_pure_rs::FastText;

const MODEL_PATH: &str = "tests/fixtures/lid.176.ftz";

fn load_model() -> FastText {
   FastText::load(MODEL_PATH).expect("failed to load test model")
}

/// Benchmark prediction latency across different input lengths.
fn bench_predict_by_input_length(c: &mut Criterion) {
   let model = load_model();

   let inputs: &[(&str, &str)] = &[
      ("short (2 chars)", "hi"),
      (
         "medium (90 chars)",
         "This is a medium length sentence that would be typical of user input in a real application",
      ),
      (
         "long (800 chars)",
         "This is a much longer string that simulates a paragraph of text. \
          It contains multiple sentences and various words that the model needs to \
          tokenize and look up in the dictionary. The purpose is to measure how \
          prediction latency scales with input length. In a real application this \
          might be a product review, a social media post, or a news article excerpt. \
          We want to ensure that longer inputs do not cause disproportionate slowdowns \
          compared to shorter inputs. The model should handle this efficiently by \
          averaging the word embeddings and running the softmax or hierarchical softmax \
          prediction step. Let us see how it performs with this roughly thousand \
          character input string that exercises many different dictionary lookups \
          and subword hash computations across a variety of common English words.",
      ),
   ];

   // Pre-flight: verify predictions succeed before entering timed loops
   for (_name, text) in inputs {
      model
         .predict(text, 1, 0.0)
         .expect("pre-flight prediction failed");
   }

   let mut group = c.benchmark_group("predict/input_length");
   for (name, text) in inputs {
      group.bench_with_input(BenchmarkId::new("k=1", name), text, |b, text| {
         b.iter(|| model.predict(text, 1, 0.0));
      });
   }
   group.finish();
}

/// Benchmark how prediction latency scales with the number of
/// requested results (k).
fn bench_predict_by_top_k(c: &mut Criterion) {
   let model = load_model();
   let text = "This is a test sentence in English";

   // Pre-flight: verify the highest-k prediction succeeds
   model
      .predict(text, 176, 0.0)
      .expect("pre-flight prediction failed");

   let mut group = c.benchmark_group("predict/top_k");
   for k in [1, 5, 10, 50, 176] {
      group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
         b.iter(|| model.predict(text, k, 0.0));
      });
   }
   group.finish();
}

/// Benchmark prediction across several languages to verify consistent
/// performance regardless of the input script.
fn bench_predict_by_language(c: &mut Criterion) {
   let model = load_model();

   let inputs: &[(&str, &str)] = &[
      ("english", "This is a test sentence in English"),
      ("french", "Ceci est une phrase en francais"),
      ("german", "Dies ist ein Satz auf Deutsch"),
      ("japanese", "これは日本語のテストです"),
      ("arabic", "هذه جملة اختبار باللغة العربية"),
   ];

   // Pre-flight: verify predictions succeed before entering timed loops
   for (_name, text) in inputs {
      model
         .predict(text, 1, 0.0)
         .expect("pre-flight prediction failed");
   }

   let mut group = c.benchmark_group("predict/language");
   for (name, text) in inputs {
      group.bench_with_input(BenchmarkId::new("k=1", name), text, |b, text| {
         b.iter(|| model.predict(text, 1, 0.0));
      });
   }
   group.finish();
}

/// Benchmark model loading time (cold start).
fn bench_model_load(c: &mut Criterion) {
   // Pre-flight: verify model loads successfully
   FastText::load(MODEL_PATH).expect("pre-flight model load failed");

   c.bench_function("model_load", |b| {
      b.iter(|| FastText::load(MODEL_PATH));
   });
}

/// Benchmark sentence vector computation.
fn bench_sentence_vector(c: &mut Criterion) {
   let model = load_model();
   let text = "This is a test sentence in English";

   // Pre-flight: verify sentence vector succeeds
   model
      .get_sentence_vector(text)
      .expect("pre-flight sentence vector failed");

   c.bench_function("sentence_vector", |b| {
      b.iter(|| model.get_sentence_vector(text));
   });
}

criterion_group!(
   benches,
   bench_predict_by_input_length,
   bench_predict_by_top_k,
   bench_predict_by_language,
   bench_model_load,
   bench_sentence_vector,
);
criterion_main!(benches);
