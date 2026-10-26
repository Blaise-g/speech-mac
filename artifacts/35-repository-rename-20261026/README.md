# Repository rename evidence

On 2026-10-26, renamed `Blaise-g/kokoro-mac` to `Blaise-g/speech-mac` from clean starting commit `d01a0d0df25ef8f64b0aa728b39765d4ed71abd3`.

Preflight confirmed target `gh api repos/Blaise-g/speech-mac` returned HTTP 404, owner `Blaise-g`, private visibility and admin permission. No explicit `remote.origin.pushurl` existed. Executed:

```sh
gh repo rename speech-mac --repo Blaise-g/kokoro-mac --yes
git remote set-url origin https://github.com/Blaise-g/speech-mac.git
```

The before/after JSON records come from `gh api repos/Blaise-g/<name>` with only the recorded fields selected, `gh api repos/Blaise-g/<name>/git/ref/heads/main`, and `gh api 'repos/Blaise-g/<name>/pulls?state=open&per_page=100'` selecting number, title, head and base. Both PR censuses were empty. Repository ID `1415715156`, node ID `R_kgDOVGIZVA`, permissions, private visibility, merge settings and default branch remained identical. Main remained `67e66a3486c68aa2ad2c0ce33b0e01eda8d12669`.

Both refs files were captured with:

```sh
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' ls-remote origin
diff -u artifacts/35-repository-rename-20261026/before-refs.txt artifacts/35-repository-rename-20261026/after-refs.txt
```

The diff was empty, including main and historical refs. `after-origin.txt` records `git remote -v`; fetch and push use the new URL. `after-ci.json` records `gh api repos/Blaise-g/speech-mac/contents/.github/workflows/ci.yml` selecting path, blob SHA, size and URL. Its blob matches the unchanged main commit via `git rev-parse 67e66a3486c68aa2ad2c0ce33b0e01eda8d12669:.github/workflows/ci.yml`: `40a5aff5005739e05da9f6ce82fd2d8595ec96d4`. The unpublished feature branch has its earlier uv setup addition (blob `7c48ea23bef63c3925b886c35363522f595abc05`); this slice does not change either workflow. The workflow remains unchanged: PRs and main pushes run job `check` on `macos-15` with `make check`.

Tests added: explicitly empty; this slice changes repository administration and documentation only. No model/device tests, benchmarks or audio playback were performed for this slice. Ticket 34's numerical/device acceptance remains the runtime evidence; user listening confirmation remains pending. No branch push or PR was performed. Final whole-branch review remains required before those actions.
