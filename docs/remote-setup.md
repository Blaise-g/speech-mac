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

The requested publication changes the repository to public after the setup and documentation review. The private-plan limitations above are dated historical evidence. Public visibility permits additional protection features but does not enable any of them; this follow-up does not change branch protections, rulesets or required checks. The coordinator records the live visibility and CI readback after publication.

The setup source and final High review are complete, and PR4's source check passed. The agent environment rejected the requested public visibility/description command before execution because it requires approval while the session uses `AskForApproval=Never`. A live readback still reports private visibility and the previous description. Public visibility, the new description and anonymous-access verification remain pending; the user has the exact manual command. No protection/ruleset setting was changed.
