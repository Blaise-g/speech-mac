# Coding standards

- Write runtime inference in readable C17 and use the repository `.clang-format` rules.
- Keep ownership and lifetime visible at interfaces. Document borrowed data and transfers of ownership where they are not obvious from the function contract.
- Keep platform calls behind narrow interfaces when that makes CPU reference checks possible without changing inference behavior.
- Add runtime dependencies only when they serve an accepted requirement; Python remains a reference and conversion tool.

Use antirez's [qwen-asr](https://github.com/antirez/qwen-asr) and [voxtral.c](https://github.com/antirez/voxtral.c) as continuing architectural references for readable, modular C inference, explicit weight and numerical-operation boundaries, and platform acceleration kept behind narrow seams. Reuse their structure, not their ASR algorithms or streaming-specific KV-cache and rollback designs: Phonon's graph is TTS-specific. Keep the shipped Phonon inference path in native C; This migration uses Accelerate/NEON on the CPU, with no Metal path. Keep Python and Rust in development reference tooling; Core ML is outside the shipped inference runtime.

The compiler flags, formatter, static analyzer, and check order are defined in the Makefile.
