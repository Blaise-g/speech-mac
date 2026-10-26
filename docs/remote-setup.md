# Remote setup record

The repository is public at [`Blaise-g/speech-mac`](https://github.com/Blaise-g/speech-mac). Origin fetch and push use `https://github.com/Blaise-g/speech-mac.git`. Readbacks on 2026-10-26 confirmed repository ID `1415715156`, default branch `main`, and description "Native C17 Phonon 40M English text-to-speech for Apple Silicon macOS". Authenticated and anonymous API access confirmed public visibility; anonymous Git and raw README access also succeeded.

Merge commits only and automatic source-branch deletion are enabled. Main is unprotected and the ruleset list is empty. Publication enabled no branch protections, rulesets or required checks. The repository is solo-maintained; contribution policy does not require a second reviewer. See [contributing](../CONTRIBUTING.md).

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs `make check` on `macos-15` for pull requests and pushes to `main`. Its job status check is `check`. [PR4](https://github.com/Blaise-g/speech-mac/pull/4) merged the setup source after final High review, and [main CI](https://github.com/Blaise-g/speech-mac/actions/runs/40346286812) passed at `50baa092d6c301cd12eee6d29d8f1b6f32e4af06`. This is dated setup evidence, not validation of later commits.

The workflow does not fetch ignored model weights. Local model-backed checks require the assets described in [README.md](../README.md); applicable C tests report explicit skips without them. Maintained locked reference commands are in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Historical records

The [archived setup record](https://github.com/Blaise-g/speech-mac/blob/3434f73aeada7e1818feb11cf4815b773a21188b/docs/remote-setup.md) retains the initial private setup, provider plan restrictions, bootstrap CI, repository rename, publication procedure and original command/readback evidence. The [rename readbacks](../artifacts/35-repository-rename-20261026/README.md) remain unchanged. Historical repository names and private visibility in those records describe their original events.
