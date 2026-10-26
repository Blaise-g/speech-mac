# Contributing

`main` is the integration branch and pull request target. The private GitHub repository allows merge commits only and deletes a pull request's source branch after merge. GitHub cannot enable branch protection or required checks for this private repository on its current plan, so merge a pull request only after the CI job named `check` succeeds. The repository is solo-maintained and does not require a second reviewer. See [remote setup](docs/remote-setup.md) for the provider readback and limits.

Before committing, run `make format`, then `make check`, and resolve failures. The repository-local pre-commit hook repeats the format check, lint, typecheck, and tests against the staged snapshot. Use `make test-one TEST=tests/test_name.c` for a focused C test. The C suite includes checked-in fixtures; model-backed parity and synthesis checks skip when the ignored model artifacts are absent. Follow the model download and conversion steps in [README.md](README.md), then rerun `make check` to run those checks. CI does not download ignored model artifacts.

Run development reference checks with the locked Phonon project:

```sh
uv run --project reference/phonon --offline --no-sync --locked python -m unittest discover -s reference/phonon/tests
make phonon-reference-check
make phonon-native-check
```

The latter two commands require the pinned local assets described in [reference setup](reference/phonon/README.md). `make check` also checks the maintained source/API/build surface for removed dependencies and commands.

Use concise conventional commit subjects such as `feat: add speech frontend`, and include `Ticket: <NN>-<slug>` in every implementation commit. Complete an independent review before committing. Update the ticket's status and verification evidence as work progresses. A local hook can be bypassed; `.github/workflows/ci.yml` runs `make check` on pull requests and pushes to `main`. GitHub currently cannot enforce that check on this private repository, so verify it before merging.
