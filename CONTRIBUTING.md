# Contributing

`main` is the integration branch and pull request target. Remote branch protection and CI status have not been configured because remote setup was not authorized.

Before committing, run `make format`, then `make check`, and resolve failures. The repository-local pre-commit hook repeats the format check, lint, typecheck, and tests against the staged snapshot. Use `make test-one TEST=tests/test_name.c` for a focused test. The current test suite is empty until behavior is implemented; the full check command reports that state explicitly.

Use concise conventional commit subjects such as `feat: add phoneme frontend`, and include `Ticket: <NN>-<slug>` in every implementation commit. Complete an independent review before committing. Update the ticket's status and verification evidence as work progresses. A local hook can be bypassed; CI runs the full check sequence on pushes to `main` and pull requests.
