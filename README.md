# fasttext-pure-rs

Pure-Rust fastText inference engine. Loads `.bin` and `.ftz` model files
and runs prediction with zero C/C++ dependencies.

## Scope

This library is **inference only** -- it loads and runs predictions on
pre-trained fastText models. Training, autotuning, and model
quantization are out of scope. Only binary format version 12 is
supported (the standard for all modern fastText models).

## No-I/O Design

The primary loading method is `load_from_reader`, which accepts any
`Read` implementor. The library itself never opens files or makes
network calls -- callers control all I/O and pass bytes in. This
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

### Testing

```sh
cargo test
```

## License

MIT
