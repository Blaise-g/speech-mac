# 35: Rename GitHub repository last

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User clarification dated 2026-10-25: subsequent implementation, fixes and ticket reviews use implement-lite / gpt-6.1-sol / low, overriding the repository model/effort defaults. Final whole-branch review uses gpt-6.1-sol / high. Actual earlier Luna/low execution records remain historical evidence.
- Consumer census: local API consumers are CLI/stream/benchmarks/tests. Nearby repositories and queried upstream PRs revealed no additional client. No UI exists. Installed `/opt/homebrew/bin/kokoro` is external and stays untouched. Census cannot rule out undiscoverable private callers. Recheck open PRs before removal/rename. Upstream PR209 context+1 discrepancy gates oracle reconciliation; PR186 Mimi Q8 was slower with quality tradeoff and is excluded.

## Footprint

GitHub repo identity; common `.git/config` shared by worktrees; origin fetch/push URLs, maintained links/admin docs/tracker. No model/device access. Per-command gh credential helper may avoid inherited helper hangs without global edits: `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' ls-remote origin refs/heads/main`.

## What to build

Rename the GitHub repository, update the origin URL, and verify repository identity, history, permissions, URLs, and CI configuration after ticket34 acceptance and its changed-path review; verify new-URL `ls-remote` and open-PR census.

Approved administrative sequence: `gh repo rename speech-mac --repo Blaise-g/kokoro-mac --yes`, then `git remote set-url origin https://github.com/Blaise-g/speech-mac.git`; inspect pushurl and replace old explicit push URL if present. After ticket34 acceptance and changed-path review, complete this ticket and its changed-path review. Run-level whole-branch review follows ticket35 and precedes the authorized push and PR. No merge/deploy or local root move.

## Blocked by

34-migration-acceptance

Worker Profile: medium; implement-lite / gpt-6.1-sol / low (user clarification 2026-10-25)

## Acceptance criteria

- [x] Rename the GitHub repository, update the origin URL, and verify repository identity, history, permissions, URLs, and CI configuration after the native migration is accepted
- [x] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: done

## Implementation and verified completion — 2026-10-26

- Wave base: `d01a0d0df25ef8f64b0aa728b39765d4ed71abd3`; branch `feat/phonon-migration`, existing run worktree. Worker `/root/35_rename_github`, attempt 1, implement-lite / GPT-6.1 Sol / low; formal repair count 0.
- Result: `b446b399cb77b9e6e32ef5d1b00eb6dfbd0a2b98`, correct Ticket35 trailer and clean worktree. Canonical atomic `35-rename-github.report.md` is done/attempt 1 in the run Briefs directory. Twelve changed paths comprise eleven sanitized administrative evidence files and `docs/remote-setup.md`; no runtime, tests, reference, vendor, fixtures, budgets, lock or workflow change.

After fresh preflight, the worker executed the approved `gh repo rename speech-mac --repo Blaise-g/kokoro-mac --yes` followed by `git remote set-url origin https://github.com/Blaise-g/speech-mac.git`. Origin has no explicit push URL override; both effective directions now use the new URL. Target absence and no explicit push URL were observed directly in the worker's preflight and recorded narratively, without separate raw preflight files. No token material, global Git configuration, local root move, system audio changes or model/device work was involved.

Before/after evidence preserves repository ID1415715156/node R_kgDOVGIZVA, owner/admin permissions, private visibility, default main SHA `67e66a3486c68aa2ad2c0ce33b0e01eda8d12669`, complete advertised refs, merge-only settings and branch deletion policy. Both open-PR censuses were empty. Remote main's CI blob `40a5aff5005739e05da9f6ce82fd2d8595ec96d4` matches the original main commit. The unpublished feature workflow's earlier uv setup blob `7c48ea23bef63c3925b886c35363522f595abc05` remains distinct;35 does not change either configuration. Maintained remote documentation links the new identity; historical bootstrap URLs remain attributed.

The worker ran format→check twice, the second after the final documentation correction; both passed. Its staged-snapshot commit hook passed. Tests added are explicitly empty for administration/documentation only. Ordinary absent-fixture skips and seven existing analyzer warnings remain explicit; ticket34's actual-model/device acceptance is unchanged. Independent GPT-6.1 Sol/low SingleReviewer examined all twelve paths through all six lenses with no findings or required fixes. Precommit check/commit obligations and narrative-only preflight evidence were accurately marked unverified in that review; the subsequent report/checks complete the former. One transient reviewer capacity error recovered using the same model/session.

The Coordinator independently ran `/private/tmp/speech-mac-35-coordinator-verify.py`, exit 0: exact twelve-path footprint, Ticket trailer and clean status; live new-repository API identity/settings/permissions, unchanged main and refs, effective origin directions with no push override, empty current PR census, unchanged main CI blob versus unpublished feature blob, and the old URL redirect to the same new repository all pass. The initial private assertion incorrectly compared old/new main-object URL strings without allowing the intended repository-name change; it was corrected to allow only that URL substitution, and retained separately. This was a verifier error, not a product or repository failure. Root format→full canonical check passed; unchanged C means prior actual-model/device gates remain current. Logs: `/private/tmp/speech-mac-35-coordinator-verify.log`, `speech-mac-35-root-format.log`, `speech-mac-35-root-check.log`.

Ticket35 is verified at `b446b399cb77b9e6e32ef5d1b00eb6dfbd0a2b98`. No Escalation or missing administrative check remains. All approved tickets19–35 are verified; after this reviewed completion writeback is committed, freeze the branch and perform the mandatory whole-branch GPT-6.1 Sol/high review, including earlier Luna-authored code. Only after triage and any accepted fixes may the authorized one branch push/one PR/CI watch proceed. No push, PR, merge or deployment has occurred. Listening remains PENDING user confirmation and acoustic onset unknown.
