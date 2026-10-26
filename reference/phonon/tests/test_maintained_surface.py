"""Ticket 31 removal contract, limited to maintained runtime/build surfaces."""
import os
from pathlib import Path
import re
import subprocess
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[3]


class MaintainedSurface(unittest.TestCase):
    def test_installed_assets_leave_checkout_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                   "XDG_CONFIG_HOME": directory}
            def git(*args):
                return subprocess.check_output(["git", "-c", "core.excludesFile=" + os.devnull,
                    *args], cwd=root, env=env, text=True).strip()
            git("init", "-q")
            (root / ".gitignore").write_text((ROOT / ".gitignore").read_text())
            git("add", ".gitignore")
            for name in ("models/phonon-40m-en/model.gguf", "voices/default.safetensors",
                         "voices/alice.safetensors"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"installed asset")
                self.assertEqual(git("check-ignore", name), name)
            self.assertEqual(git("ls-files", "--others", "--exclude-standard"), "")
            self.assertEqual(git("diff", "--name-only"), "")

    def test_no_legacy_runtime_or_public_identifiers(self):
        removed = ("main.c", "cli.c", "kokoro.h", "kokoro.c", "model_pack.c", "synthesis.c",
                   "english_frontend.c", "ipa_tokenizer.c", "voice_artifact.c", "stream.c")
        for name in removed:
            self.assertFalse((ROOT / "src" / name).exists(), name)
        for path in (ROOT / "src").iterdir():
            if path.suffix not in (".h", ".c"):
                continue
            source = path.read_text()
            self.assertIsNone(re.search(r"\b(?:Kokoro[A-Z]\w*|KOKORO_\w*|kokoro_\w*)\b", source), path)
            self.assertIsNone(re.search(r"(?:python|espeak|CoreML|\.py[\"'])", source), path)

    def test_no_old_reference_or_active_benchmark(self):
        for name in ("oracle.py", "convert.py", "assets.py", "pack.py", "voice_artifact.py",
                     "uv.lock", "pyproject.toml"):
            self.assertFalse((ROOT / "reference" / name).exists(), name)
        for directory in ("fixtures", "tests"):
            self.assertFalse(any((ROOT / "reference" / directory).rglob("*.py")))
            self.assertFalse(any(p.is_file() for p in (ROOT / "reference" / directory).glob("*")))
        for path in (ROOT / "benchmarks").iterdir():
            if path.name in {"phonon_baseline.c", "phonon_queue.c"}:
                continue
            self.assertEqual(path.suffix, ".md", path)
        build = (ROOT / "Makefile").read_text()
        self.assertNotIn("libkokoro", build)
        self.assertNotIn("/kokoro", build)
        self.assertNotIn("LEGACY_SOURCES", build)

    def test_built_command_identity_and_migration_errors(self):
        binary = ROOT / os.environ.get("BUILD_DIR", ".build") / "speech-mac"
        if not binary.exists():
            self.skipTest("built CLI unavailable; run make build for executable census")
        self.assertEqual(subprocess.check_output([binary, "--version"], text=True), "speech-mac 0.1.0\n")
        for option in ("--ipa", "--speed", "-s"):
            result = subprocess.run([binary, "say", option, "1"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("removed Kokoro option", result.stderr)
            self.assertIn("ordinary English", result.stderr)
        symbols = subprocess.check_output(["nm", "-g", binary], text=True)
        self.assertNotIn("kokoro_", symbols)
        libraries = subprocess.check_output(["otool", "-L", binary], text=True).lower()
        for dependency in ("python", "espeak", "coreml", "rust"):
            self.assertNotIn(dependency, libraries)
