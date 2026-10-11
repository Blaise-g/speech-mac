# Kokoro native for macOS

Native C inference for Kokoro-82M v1.0, targeting Apple Silicon macOS. The initial runtime will use Accelerate. Python is reserved for reference checks and weight conversion; it will not be part of the shipped inference path.

## Prerequisites

- An Apple Silicon Mac running macOS with Xcode Command Line Tools (`clang`, `make`, and `xcrun`).
- Git.

`uv` will manage Python reference and conversion tools when those tools are added. eSpeak NG will be required when the English text frontend is implemented. Neither is needed by this foundation.

## Development and checks

Run `make` to list the available commands. Use `make setup` once in a clone to install the repository-local pre-commit hook, and `make dev` to build the development CLI. Run `make format` before committing, then `make check` for the full local check sequence. The hook runs `make check-fast` against the staged snapshot.

The repository currently has no runtime C sources or behavior. Until implementation adds them, build, format, lint, and typecheck commands report an explicit skip, and `make test` reports an empty suite. Once added, focused C tests use the `tests/test_*.c` naming pattern and can be run with `make test-one TEST=tests/test_name.c`. `make check` covers format check, lint, typecheck, build, and the full test suite; reference comparisons, audio listening, and performance measurements will be documented with the implementation that introduces them.

## Project guidance

- [Agent instructions](AGENTS.md) point to the coding standards, domain decisions, and local tracker.
- [Contributing](CONTRIBUTING.md) describes branch, review, commit, and verification rules.
- [Local tracker](.scratch/kokoro-native/README.md) defines the authoritative spec and ticket locations.
