from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "reference/phonon/normalization/fixtures.jsonl"
ORACLE = ROOT / "reference/phonon/normalization-oracle/Cargo.toml"
PROBE = ROOT / ".build/english-normalize-probe"


def main() -> int:
    cases = [json.loads(line) for line in FIXTURES.read_text(encoding="utf-8").splitlines()]
    source = "".join(case["input"] + "\n" for case in cases).encode("utf-8")
    env = os.environ.copy()
    env.setdefault("CARGO_HOME", str(ROOT / ".build/reference-toolchain/cargo"))
    env.setdefault("RUSTUP_HOME", str(ROOT / ".build/reference-toolchain/rustup"))
    env.setdefault("CARGO_TARGET_DIR", str(ROOT / ".build/normalization-oracle-target"))
    cargo = Path(env.get("CARGO", ROOT / ".build/reference-toolchain/cargo/bin/cargo"))
    oracle = subprocess.run(
        [str(cargo), "run", "--quiet", "--manifest-path", str(ORACLE), "--offline", "--locked"],
        cwd=ROOT,
        env=env,
        input=source,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.decode("utf-8").splitlines()
    subprocess.run(["make", "english-normalization-probe"], cwd=ROOT, check=True)
    native = subprocess.run(
        [str(PROBE)], cwd=ROOT, input=source, stdout=subprocess.PIPE, check=True
    ).stdout.decode("utf-8").splitlines()
    expected = [case["expected"] for case in cases]
    for i, (case, want, rust, c) in enumerate(zip(cases, expected, oracle, native)):
        if rust != want or c != want:
            raise AssertionError(
                f"case {i} {case['input']!r}: fixture={want!r}, upstream={rust!r}, native={c!r}"
            )
    if not (len(expected) == len(oracle) == len(native)):
        raise AssertionError("oracle or native output count differs from fixture count")
    print(f"{len(cases)} pinned English normalization cases match upstream and native C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
