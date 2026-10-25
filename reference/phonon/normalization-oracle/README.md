# Pinned English normalization oracle

This development-only executable copies the text preprocessing modules from `gradium-ai/xn-ptts` revision `ac0e1524cfd9e71d57aadca0de542796f942d775`. It has no Cargo dependencies. The copied upstream files are under Apache-2.0 OR MIT; both license texts are included here. The product runtime does not link this oracle.

It reads one UTF-8 line per input and writes the pinned upstream default-English normalization for that line. Build and run with the isolated toolchain prepared at `.build/reference-toolchain`:

```sh
CARGO_HOME="$PWD/.build/reference-toolchain/cargo" \
RUSTUP_HOME="$PWD/.build/reference-toolchain/rustup" \
CARGO_TARGET_DIR="$PWD/.build/normalization-oracle-target" \
.build/reference-toolchain/cargo/bin/cargo run --manifest-path reference/phonon/normalization-oracle/Cargo.toml --offline --locked
```

The oracle modules intentionally contain upstream's other language and optional-rule code because the English implementation shares helpers with it; this command selects only `Lang::En` and `Rules::DEFAULT`. Reproduction requires the isolated official Rust 1.99.1 toolchain (rustc `58cf8f9c3`, installed under `.build/reference-toolchain` in the implementation worktree); no global rustup configuration is used. Set `CARGO`, `CARGO_HOME`, and `RUSTUP_HOME` to an equivalent isolated install if reproducing elsewhere. The generated native Unicode property tables use the same toolchain: `char::is_alphanumeric` for URL characters and `char::is_alphabetic` for numeric hyphen suffix boundaries.
