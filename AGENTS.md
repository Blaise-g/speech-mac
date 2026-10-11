# Agent instructions

Read [CODING_STANDARDS.md](CODING_STANDARDS.md) before changing C code. Read [GLOSSARY.md](GLOSSARY.md) and [docs/adr/](docs/adr/) when changing domain terms or architecture.

The authoritative local tracker is `.scratch/kokoro-native/spec.md` plus `.scratch/kokoro-native/issues/<NN>-<slug>.md`; see its [rules](.scratch/kokoro-native/README.md). Keep tracker files committed. Feature implementation starts from a coordinator-approved Run Plan.

Implementation workers use GPT-6 Luna at xhigh reasoning with the `implement-lite` workflow. Before editing, record the starting commit. Add defect-revealing tests with executable behavior, and keep the suite explicitly empty when no behavior exists. Before a commit, run `make format`, then `make check`; an independent reviewer checks the changed paths before commit. Use a `Ticket: <NN>-<slug>` commit trailer.

Use [CONTRIBUTING.md](CONTRIBUTING.md) for the canonical commands and completion rules.

Use `uv` for all Python project tooling, dependency resolution, and execution; follow the relevant checked-in `uv.lock` and run project Python commands through `uv`.
