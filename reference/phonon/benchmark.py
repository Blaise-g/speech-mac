"""Native Phonon benchmark orchestration; run with the checked-in uv project."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = {
    "short": "Hello there. I am ready to help you today.",
    "structured": "Meeting notes for October 26, 2026. First, review the budget of $1,250. Next, email alex@example.com before 3:30 pm. Finally, confirm that all three tasks are complete.",
    "long": "The morning train moved slowly through the valley. Outside the window, fields gave way to small towns and quiet roads. A passenger opened a notebook and began to describe the journey. She wrote about the changing light, the distant hills, and the people waiting on each platform. At the next stop, a family boarded with a picnic basket. They talked about the walk they planned to take beside the river. By noon the clouds had cleared, and sunlight filled the carriage. The train continued toward the coast, where the passengers would spend the afternoon exploring a new city.",
}
THREAD_ENV = ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "ACCELERATE_NEW_LAPACK")
FLAGS = ["-Isrc", "-std=c17", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-O2"]

def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def validate(report: dict) -> None:
    """Reject inconsistent units, missing repetitions and fabricated timing labels."""
    if report["schema"] != 1 or report["acoustic_onset_s"] is not None:
        raise ValueError("schema/acoustic onset")
    if report["provenance"]["source_dirty"] or not report["provenance"]["source_commit"]:
        raise ValueError("final report must identify committed source")
    for key in ("pack_sha256", "voice_sha256"):
        if len(report["provenance"][key]) != 64:
            raise ValueError(key)
    seen = set()
    for r in report["runs"]:
        key = (r["workload"], r["mode"], r["repetition"])
        if key in seen:
            raise ValueError("duplicate repetition")
        seen.add(key)
        for field in ("first_pcm_s", "synthesis_s", "cpu_s", "engine_prepare_s", "voice_open_s"):
            if not math.isfinite(r[field]) or r[field] <= 0:
                raise ValueError(field)
        if any(type(r[k]) is not int or r[k] <= 0 for k in ("samples", "frames", "chunks", "rss_high_water_bytes")):
            raise ValueError("positive integral counts/bytes")
        if r["first_pcm_s"] > r["synthesis_s"] or r["samples"] != r["frames"] * 1920 or not 1 <= r["frames"] <= 8192 or r["chunks"] < 1:
            raise ValueError("PCM/count invariant")
        if r["rss_high_water_bytes"] <= 0 or r["output_duration_s"] != r["samples"] / 24000:
            raise ValueError("memory/duration")
        if not math.isclose(r["rtf"], r["synthesis_s"] / r["output_duration_s"], rel_tol=1e-12):
            raise ValueError("RTF")
        if r["profiled"] or r["consumer"] != "synchronous_memory_copy":
            raise ValueError("baseline timing scope")
    expected = {(w, m, i) for w in WORKLOADS for m in ("fresh_process", "warm_resident") for i in range(report["repeats"])}
    if seen != expected or report["repeats"] < 5:
        raise ValueError("missing workload/mode/repetition")
    for w in WORKLOADS:
        if len({(r["samples"], r["pcm_f32_sha256"]) for r in report["runs"] if r["workload"] == w}) != 1:
            raise ValueError("request reset/count mismatch")

def build(profile: bool) -> Path:
    binary = ROOT / ".build" / ("phonon-baseline-profile" if profile else "phonon-baseline")
    sources = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src").glob("*.c") if p.name != "speech_main.c")
    subprocess.run(["xcrun", "clang", *FLAGS, *(["-DSPEECH_PROFILE"] if profile else []), "benchmarks/phonon_baseline.c", *sources, "-framework", "AudioToolbox", "-framework", "Accelerate", "-o", str(binary)], cwd=ROOT, check=True)
    return binary

def execute(binary: Path, pack: Path, voice: Path, text: Path, count: int, wav: Path | None) -> list[dict]:
    output = subprocess.check_output([str(binary), str(pack), str(voice), str(text), str(count), str(wav) if wav else "-"], cwd=ROOT, text=True)
    return [json.loads(line) for line in output.splitlines()]

def run(args: argparse.Namespace) -> None:
    if not 5 <= args.repeats <= 19:
        raise SystemExit("--repeats must be between 5 and 19 (plus one resident warmup)")
    if command("git", "status", "--porcelain", "--untracked-files=normal"):
        raise SystemExit("Commit reviewed benchmark code before final measurements")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pack, voice = args.pack.resolve(), args.voice.resolve()
    model_dir = Path(os.environ["PHONON_MODEL_DIR"])
    assets = json.loads((model_dir / "local-verification.json").read_text())
    for asset in assets["files"]:
        if digest(model_dir / asset["filename"]) != asset["sha256"]:
            raise ValueError("source model artifact hash mismatch: " + asset["filename"])
    normal, profile = build(False), build(True)
    provenance = {
        "source_commit": command("git", "rev-parse", "HEAD"), "source_dirty": False,
        "host": platform.platform(), "machine": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(command("sysctl", "-n", "hw.memsize")), "os_build": command("sw_vers"),
        "compiler": command("xcrun", "clang", "--version"), "flags": FLAGS,
        "binary_sha256": digest(normal), "profile_binary_sha256": digest(profile),
        "pack_path": str(pack), "pack_sha256": digest(pack), "voice_path": str(voice), "voice_sha256": digest(voice),
        "source_assets": assets,
        "model_revision": "cc738b7e4d182c2ba1bafb8062aa4846af000637", "quantization": {"Q8_0": 36, "BF16": 173},
        "model_id": "8aa279a6@100", "voice_model_ext": "ba7a5a45@300",
        "thread_environment": {k: os.environ.get(k) for k in THREAD_ENV},
        "generation": {"seed": 0, "language": "en", "duration_delta": 0, "temperature": .8, "lsd_steps": 1, "eos_threshold": 4, "min_eos": 0, "short_tail": 3, "normal_tail": 1, "sample_rate": 24000},
    }
    runs, profiles, auditions = [], [], []
    for name, text in WORKLOADS.items():
        text_path = out / f"{name}.txt"
        text_path.write_text(text)
        for i in range(args.repeats):
            raw = execute(normal, pack, voice, text_path, 1, None)[0]
            raw.update(workload=name, mode="fresh_process", repetition=i)
            runs.append(raw)
        wav = out / f"{name}.wav"
        warm = execute(normal, pack, voice, text_path, args.repeats + 1, wav)
        for i, raw in enumerate(warm):
            raw.update(workload=name, mode="warm_resident", repetition=i-1)
            if i:
                runs.append(raw)
            else:
                (out / f"{name}-excluded-warmup.json").write_text(json.dumps(raw, indent=2)+"\n")
        profile_wav = out / f"{name}-profile.wav"
        prof = execute(profile, pack, voice, text_path, 2, profile_wav)
        if prof[1]["pcm_f32_sha256"] != warm[-1]["pcm_f32_sha256"] or digest(profile_wav) != digest(wav):
            raise ValueError("instrumented/uninstrumented PCM differs")
        profiles.append({"workload": name, "excluded_warmup": prof[0], "measurement": prof[1]})
        auditions.append({"workload": name, "path": wav.name, "sha256": digest(wav), "text": text, "listening": "pending"})
    for r in runs:
        r.update(profiled=False, consumer="synchronous_memory_copy", output_duration_s=r["samples"]/24000)
        r["rtf"] = r["synthesis_s"] / r["output_duration_s"]
    report = {"schema": 1, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "repeats": args.repeats, "provenance": provenance, "runs": runs, "profiles": profiles, "auditions": auditions, "acoustic_onset_s": None}
    validate(report)
    (out / "raw.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# Phonon 40M native baseline", "", f"Source: `{provenance['source_commit']}`. All times are seconds; Darwin RSS high water is bytes.", "", "| Workload | Mode | First PCM median (range) | Synthesis median (range) | RTF median |", "|---|---|---|---|---|"]
    for w in WORKLOADS:
        for m in ("fresh_process", "warm_resident"):
            rows = [r for r in runs if r["workload"] == w and r["mode"] == m]
            def spread(k):
                v = [r[k] for r in rows]
                return f"{statistics.median(v):.4f} ({min(v):.4f}–{max(v):.4f})"
            lines.append(f"| {w} | {m} | {spread('first_pcm_s')} | {spread('synthesis_s')} | {statistics.median(r['rtf'] for r in rows):.4f} |")
    lines += ["", "Fresh process means a new child, with no OS cache purge. First PCM and synthesis start at the native request call, after separately recorded engine/voice preparation. Voice projection and request state initialization are inside synthesis. CPU is process user+system delta over synthesis. RSS is process lifetime high water, not current memory or a request allocation delta. The resident child excludes one recorded warmup and reuses engine/voice while each request resets state and seed. Callback copies PCM; no playback occurs.", "", "Top-level stage wall scopes are disjoint producer-thread intervals, separately compiled and measured with instrumentation. They include callee work and exclude callback/output I/O; prefill, flow head, Mimi/PCM and incremental backbone are not operator-exclusive CPU samples. Additional linear_expand_bf16, linear_expand_q8 and linear_sgemm operator scopes are nested within the top-level stages (and request preparation); never add them to stage totals. Expansion scopes include allocation and finite-value conversion, but not GEMM; GEMM scopes time only the Accelerate call. Unattributed synthesis work includes frontend/planning, voice projection and request setup/cleanup. Profile warmup is retained and excluded. Profile timing is not used in baseline medians.", "", "Audition WAVs have not been played; listening remains pending. Acoustic onset is unavailable without independent capture. Playback queue events are a separate artifact and are never inferred from PCM or fake consumer timings.", "", "## Separate stage profile", "", "| Workload | Prefill s | Flow head s | Mimi/PCM s | Incremental backbone s |", "|---|---|---|---|---|"]
    for p in profiles:
        stages = p['measurement']['stages']
        lines.append("| "+p['workload']+" | "+" | ".join(f"{stages[k]['seconds']:.4f}" for k in ("prefill", "flow_head", "mimi_pcm", "incremental_backbone"))+" |")
    lines += ["", "Nested linear operators (overlap stages above):", "", "| Workload | BF16 expansion s | Q8 expansion s | SGEMM s |", "|---|---|---|---|"]
    for p in profiles:
        stages = p["measurement"]["stages"]
        lines.append("| "+p["workload"]+" | "+" | ".join(f"{stages[k]['seconds']:.4f}" for k in ("linear_expand_bf16", "linear_expand_q8", "linear_sgemm"))+" |")
    (out / "report.md").write_text("\n".join(lines)+"\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--voice", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    run(p.parse_args())

if __name__ == "__main__":
    main()
