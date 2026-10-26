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

The branch-protection and repository-ruleset APIs returned GitHub's plan-limit response: this feature requires GitHub Pro for a private repository or a public repository. At that time the project kept private visibility, so GitHub could not enforce PR-only changes, prevent force-pushes or deletions, or require the CI check. The repository is solo-maintained, so its contribution policy does not require a second reviewer. The [protected branches documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches) and [ruleset availability documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets) describe the provider limits.

## CI evidence

The configured workflow is [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). It runs `make check` on `macos-15` for pull requests and pushes to `main`. GitHub emitted the job status check as `check`.

The observed bootstrap run was [run 38263195103](https://github.com/Blaise-g/kokoro-mac/actions/runs/38263195103), triggered by the `main` push of setup commit `963ce48eb1da6f6b067588b067e6d325313bcf93`. Its `check` job completed successfully. This verifies CI on the setup commit; it does not represent a run for the unpublished feature branch.

The workflow does not fetch ignored model weights. Local model-backed checks require the artifacts described in [README.md](../README.md); without them, the applicable C tests report explicit skips. The maintained locked Phonon reference commands are documented in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Repository rename on 2026-10-26

The current repository is [`Blaise-g/speech-mac`](https://github.com/Blaise-g/speech-mac), renamed from `Blaise-g/kokoro-mac`. Origin fetch and push use `https://github.com/Blaise-g/speech-mac.git`. The local checkout paths were retained.

The [before/after readbacks](../artifacts/35-repository-rename-20261026/README.md) verify the same repository ID, complete remote refs, main SHA, private visibility, permissions, merge settings and CI workflow. Both open-PR censuses were empty. The bootstrap run URL above retains its historical repository name as provenance. The rename did not push the feature branch or run feature-branch CI.

## Public setup follow-up, 2026-10-26

The repository is now public. The private-plan limitations above are dated historical evidence. Public visibility permits additional protection features but does not enable any of them; this follow-up does not change branch protections, rulesets or required checks.

The setup source and final High review are complete. [PR4](https://github.com/Blaise-g/speech-mac/pull/4) is merged, and [main CI](https://github.com/Blaise-g/speech-mac/actions/runs/40346286812) passed at `50baa092d6c301cd12eee6d29d8f1b6f32e4af06`.

The agent environment rejected the visibility/description command because it required approval while the session used `AskForApproval=Never`. The user then ran the command in Terminal. Subsequent authenticated and anonymous GitHub API readbacks confirm public visibility, repository ID `1415715156`, default branch `main`, and description “Native C17 Phonon 40M English text-to-speech for Apple Silicon macOS”. Anonymous Git access resolves HEAD and main to the merged commit above; the raw README returns HTTP 200. Merge commits only and automatic source-branch deletion remain enabled. Main remains unprotected and the ruleset list is empty; no protection setting was changed.
