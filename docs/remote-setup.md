# Remote setup record

Checked on 2026-10-12 for `Blaise-g/kokoro-mac`.

| Setting | Result | Evidence |
| --- | --- | --- |
| Visibility | Verified private; kept private | GitHub repository API readback |
| Default branch and PR target | Verified `main` | GitHub repository API; `CONTRIBUTING.md` |
| PR merge method | Verified merge commits only; GraphQL default is `MERGE` | Repository settings readback |
| Delete merged PR branches | Enabled and verified | Repository settings readback: `delete_branch_on_merge: true` |
| Main branch protection | Unavailable | `main` reports `protected: false`; branch-protection API returned HTTP 403 |
| Required CI check | Unavailable | Same plan restriction; the configured workflow emits job check `check` |
| Production branch | Not configured | No deployment workflow or separate production branch was present |

The branch-protection and repository-ruleset APIs returned GitHub's plan-limit response: this feature requires GitHub Pro for a private repository or a public repository. The project keeps its current private visibility, so GitHub cannot enforce PR-only changes, prevent force-pushes or deletions, or require the CI check. The repository is solo-maintained, so its contribution policy does not require a second reviewer. The [protected branches documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches) and [ruleset availability documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets) describe the provider limits.

## CI evidence

The configured workflow is [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). It runs `make check` on `macos-15` for pull requests and pushes to `main`. GitHub emitted the job status check as `check`.

The observed bootstrap run was [run 38263195103](https://github.com/Blaise-g/kokoro-mac/actions/runs/38263195103), triggered by the `main` push of setup commit `963ce48eb1da6f6b067588b067e6d325313bcf93`. Its `check` job completed successfully. This verifies CI on the setup commit; it does not represent a run for the unpublished feature branch.

The workflow does not fetch ignored model weights. Local model-backed checks require the artifacts described in [README.md](../README.md); without them, the applicable C tests report explicit skips. The standalone locked comparison test command and its scope are documented in [CONTRIBUTING.md](../CONTRIBUTING.md).
