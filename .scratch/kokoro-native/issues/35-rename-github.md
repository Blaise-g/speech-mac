# 35: Rename GitHub repository last

## Parent

None (Requirement Source).

## Context

- Technical source: `.scratch/kokoro-native/spec.md` and its pinned primary sources.
- Background: `.scratch/kokoro-native/spec.md`.
- User approval dated 2026-10-25: “use implement lite with effort low for all”; resolved workflow/model/effort are implement-lite / gpt-6-luna / low, overriding AGENTS xhigh. Reviews also use Luna/low.
- Consumer census: local API consumers are CLI/stream/benchmarks/tests. Nearby repositories and queried upstream PRs revealed no additional client. No UI exists. Installed `/opt/homebrew/bin/kokoro` is external and stays untouched. Census cannot rule out undiscoverable private callers. Recheck open PRs before removal/rename. Upstream PR209 context+1 discrepancy gates oracle reconciliation; PR186 Mimi Q8 was slower with quality tradeoff and is excluded.

## Footprint

GitHub repo identity; common `.git/config` shared by worktrees; origin fetch/push URLs, maintained links/admin docs/tracker. No model/device access. Per-command gh credential helper may avoid inherited helper hangs without global edits: `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' ls-remote origin refs/heads/main`.

## What to build

Rename the GitHub repository, update the origin URL, and verify repository identity, history, permissions, URLs, and CI configuration after ticket34 acceptance and its changed-path review; verify new-URL `ls-remote` and open-PR census.

Approved administrative sequence: `gh repo rename speech-mac --repo Blaise-g/kokoro-mac --yes`, then `git remote set-url origin https://github.com/Blaise-g/speech-mac.git`; inspect pushurl and replace old explicit push URL if present. After ticket34 acceptance and changed-path review, complete this ticket and its changed-path review. Run-level whole-branch review follows ticket35 and precedes the authorized push and PR. No merge/deploy or local root move.

## Blocked by

34-migration-acceptance

Worker Profile: medium

## Acceptance criteria

- [ ] Rename the GitHub repository, update the origin URL, and verify repository identity, history, permissions, URLs, and CI configuration after the native migration is accepted
- [ ] Complete the approved slice checks and keep unavailable model/device checks explicit.

Status: ready-for-agent
