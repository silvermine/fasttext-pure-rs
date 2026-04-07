# fasttext-pure-rs

Pure-Rust fastText inference engine. Loads `.bin` and `.ftz` model files
and runs prediction with zero C/C++ dependencies.

## Why

Facebook's [fastText](https://fasttext.cc/) C++ library does not
cross-compile cleanly for Android and iOS without maintaining a custom
fork. This crate provides a pure-Rust implementation of the
inference-only path, compiling for all targets supported by the Rust
toolchain — including Android, iOS, macOS, Windows, and WebAssembly.

The primary use case is language identification via the `lid.176.ftz`
model (~917 KB, 176 languages), but the library supports any fastText
model for supervised text classification.

## Quick Start

Add `fasttext-pure-rs` to your `Cargo.toml`:

```toml
[dependencies]
fasttext-pure-rs = "0.1.0"
```

Load a model and predict:

```rust,no_run
use fasttext_pure_rs::FastText;

fn main() {
   let model = FastText::load("lid.176.ftz").unwrap();
   let predictions = model.predict("hello world", 1, 0.0).unwrap();

   for p in &predictions {
      println!("{}: {:.4}", p.label, p.probability);
   }
}
```

### Loading from Embedded Bytes

For Tauri or other embedded applications, load from any `Read` source:

```rust,no_run
use fasttext_pure_rs::FastText;

fn main() {
   let bytes = include_bytes!("path/to/lid.176.ftz");
   let model = FastText::load_from_reader(std::io::Cursor::new(bytes)).unwrap();
   let predictions = model.predict("bonjour le monde", 1, 0.0).unwrap();
   println!("{}", predictions[0].label); // __label__fr
}
```

### Sentence Embeddings

Compute a sentence's embedding vector:

```rust,no_run
use fasttext_pure_rs::FastText;

fn main() {
   let model = FastText::load("lid.176.ftz").unwrap();
   let vector = model.get_sentence_vector("hello world").unwrap();
   println!("{:?}", vector); // 16-dimensional vector for lid.176.ftz
}
```

## Supported Model Formats

| Format | Description                        | Example                  |
| ------ | ---------------------------------- | ------------------------ |
| `.bin` | Unquantized dense weight matrices  | `lid.176.bin` (~126 MB)  |
| `.ftz` | Quantized via product quantization | `lid.176.ftz` (~917 KB)  |

Both formats use fastText binary format version 12, the standard for
all modern fastText models. Older format versions are not supported.

## Obtaining the `lid.176.ftz` Model

Download the official language identification model from Facebook:

```sh
curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

This model identifies 176 languages and is ~917 KB.

## Thread Safety

`FastText` is `Send + Sync` — the loaded model is read-only and can be
shared across threads via `Arc`:

```rust,no_run
use fasttext_pure_rs::FastText;
use std::sync::Arc;

fn main() {
   let model = Arc::new(FastText::load("lid.176.ftz").unwrap());

   let handle = {
      let model = Arc::clone(&model);
      std::thread::spawn(move || {
         model.predict("hello", 1, 0.0).unwrap()
      })
   };

   let predictions = handle.join().unwrap();
   println!("{}", predictions[0].label);
}
```

## Scope

This library is **inference only** — it loads and runs predictions on
pre-trained fastText models. Training, autotuning, and model
quantization are out of scope. Only binary format version 12 is
supported (the standard for all modern fastText models).

## No-I/O Design

The primary loading method is `load_from_reader`, which accepts any
`Read` implementor. The library itself never opens files or makes
network calls — callers control all I/O and pass bytes in. This
follows the principle that libraries should be I/O-free so that the
same code works in server, Lambda, desktop, mobile, and WASM
environments without modification.

A convenience `load` method that opens a file by path is available
behind the **`std`** feature (enabled by default). Disable default
features to compile without any filesystem dependency:

```toml
[dependencies]
fasttext-pure-rs = { version = "0.1", default-features = false }
```

## Architecture

| Module                | Responsibility                                       |
| --------------------- | ---------------------------------------------------- |
| `lib.rs`              | Public API: `FastText`, `Prediction`                 |
| `error.rs`            | `Error` enum via `thiserror`                         |
| `io.rs`               | Binary format reader utilities                       |
| `args.rs`             | `ModelArgs` parsed from model header                 |
| `dictionary.rs`       | Word/subword FNV-1a hashing, tokenization            |
| `matrix.rs`           | `DenseMatrix` for `.bin` models                      |
| `quantized_matrix.rs` | `ProductQuantizer` + `QuantMatrix` for `.ftz` models |
| `model.rs`            | Inference engine: softmax, hierarchical softmax      |

## Developing

### Prerequisites

   * Rust 1.89+ (install via [rustup](https://rustup.rs/))
   * Node.js (for commitlint and markdownlint tooling)

### Setup

```sh
npm install
```

### Linting

```sh
npm run standards
```

Auto-fix (Rust only):

```sh
npm run rust:lint:fix
```

Markdownlint auto-fix:

```sh
npm run markdownlint:fix
```

### Documentation

All public items must have doc comments with code examples. Annotate
all code blocks with `no_run` to prevent `cargo test` from executing
them:

````rust
/// # Examples
///
/// ```no_run
/// use fasttext_pure_rs::FastText;
/// let model = FastText::load("model.ftz").unwrap();
/// ```
````

### Testing

```sh
cargo test
```

The test suite includes golden-output parity tests for 35 languages
using real-world text from [jw.org](https://www.jw.org/). Expected
labels and probabilities were generated by running the C++ `fasttext`
binary (`predict-prob`) and are verified for both `.ftz` and `.bin`
model formats. The `.bin` fixture (~126 MB) is tracked via Git LFS
and tests skip gracefully when LFS objects are not available.

Run aggregate accuracy validation tests (ignored by default):

```sh
cargo test -- --ignored
```

### Benchmarking

Run prediction and model-loading benchmarks with
[Criterion](https://bheisler.github.io/criterion.rs/):

```sh
cargo bench
```

This measures:

   * **Prediction latency by input length** — short (2 chars),
     medium (90 chars), and long (800 chars) strings
   * **Top-k scaling** — k = 1, 5, 10, 50, 176
   * **Cross-language consistency** — English, French, German,
     Japanese, Arabic
   * **Model load time** — cold-start from disk
   * **Sentence vector computation**

Results are written to `target/criterion/` with HTML reports. Open
`target/criterion/report/index.html` after a run to view graphs and
statistical analysis.

## License

MIT
