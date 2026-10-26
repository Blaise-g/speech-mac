# Agent instructions

Read [CODING_STANDARDS.md](CODING_STANDARDS.md) before changing C code. Read [GLOSSARY.md](GLOSSARY.md) and [docs/adr/](docs/adr/) when changing domain terms or architecture.

The authoritative local tracker is `.scratch/kokoro-native/spec.md` plus `.scratch/kokoro-native/issues/<NN>-<slug>.md`; see its [rules](.scratch/kokoro-native/README.md). Keep tracker files committed. Feature implementation starts from a coordinator-approved Run Plan.

Implementation workers use the model and reasoning effort in the coordinator-approved Run Plan or explicit user override, with the `implement-lite` workflow. The current migration uses GPT-6.1 Sol at low reasoning; final whole-branch review uses GPT-6.1 Sol at high reasoning. Before editing, record the starting commit. Add defect-revealing tests with executable behavior, and keep the suite explicitly empty when no behavior exists. Before a commit, run `make format`, then `make check`; an independent reviewer checks the changed paths before commit. Use a `Ticket: <NN>-<slug>` commit trailer.

Use [CONTRIBUTING.md](CONTRIBUTING.md) for the canonical commands and completion rules.

Use `uv` for all Python project tooling, dependency resolution, and execution; follow the relevant checked-in `uv.lock` and run project Python commands through `uv`.
