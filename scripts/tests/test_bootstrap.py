"""Ticket 36 contracts: no setup on unsupported hosts, verified reuse, safe install."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "scripts/bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
import verify_assets


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bootstrap tests ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, name="source", data=b"canonical bytes"):
        path = self.root / name
        path.write_bytes(data)
        return path

    def test_unsupported_host_does_not_start_setup(self):
        tools = self.root / "bin"
        tools.mkdir()
        (tools / "uname").write_text("#!/bin/sh\nprintf 'Linux\\n'\n")
        (tools / "uname").chmod(0o755)
        result = subprocess.run(["/bin/sh", str(ROOT / "scripts/bootstrap.sh")],
                                cwd=self.root, env={**os.environ, "PATH": str(tools)},
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Apple Silicon", result.stderr)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["bin"])

    def test_missing_prerequisites_do_not_start_uv(self):
        tools = self.root / "bin"
        tools.mkdir()
        (tools / "uname").write_text('#!/bin/sh\ncase "$1" in -s) echo Darwin;; -m) echo arm64;; esac\n')
        (tools / "uname").chmod(0o755)
        result = subprocess.run(["/bin/sh", str(ROOT / "scripts/bootstrap.sh")],
                                env={**os.environ, "PATH": str(tools)}, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("install uv", result.stderr)
        (tools / "uv").write_text('#!/bin/sh\necho unwanted >> "' + str(self.root / 'started') + '"\n')
        (tools / "uv").chmod(0o755)
        result = subprocess.run(["/bin/sh", str(ROOT / "scripts/bootstrap.sh")],
                                env={**os.environ, "PATH": str(tools)}, capture_output=True, text=True)
        self.assertIn("Xcode Command Line Tools", result.stderr)
        self.assertFalse((self.root / "started").exists())

    def pins(self):
        # The verifier's public schema and pinned revision guard remain real.
        pins = json.loads((ROOT / "reference/phonon/pins.json").read_text())
        for index, record in enumerate(pins["files"]):
            data = f"asset {index}".encode()
            record["size_bytes"] = len(data)
            record["sha256"] = hashlib.sha256(data).hexdigest()
            target = self.root / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        path = self.root / "pins.json"
        path.write_text(json.dumps(pins))
        return path, pins

    def test_all_valid_assets_reuse_without_importing_network_dependency(self):
        pins, _ = self.pins()
        with patch.object(verify_assets, "PINS", pins), patch.dict("sys.modules", {"download_assets": None}):
            bootstrap.ensure_assets(self.root)

    def test_each_pin_is_checked_and_auth_failure_is_actionable(self):
        pins_path, pins = self.pins()
        # An unavailable HF transport is the external boundary; hash checks are real.
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = lambda **kwargs: (_ for _ in ()).throw(ConnectionError("offline or gated"))
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            import download_assets
            with patch.object(verify_assets, "PINS", pins_path), patch.object(download_assets, "PINS", pins_path):
                for record in pins["files"]:
                    with self.subTest(asset=record["path"]):
                        target = self.root / record["path"]
                        original = target.read_bytes()
                        target.write_bytes(b"corrupt")
                        with self.assertRaisesRegex(RuntimeError, "hf auth login"):
                            bootstrap.ensure_assets(self.root)
                        self.assertEqual(target.read_bytes(), b"corrupt")
                        target.write_bytes(original)
                (self.root / pins["files"][0]["path"]).unlink()
                with self.assertRaisesRegex(RuntimeError, "gated terms"):
                    bootstrap.ensure_assets(self.root)

    def test_missing_asset_recovers_through_canonical_downloader(self):
        pins_path, pins = self.pins()
        cache = self.root / "hf cache"
        cache.mkdir()
        cached = {}
        for record in pins["files"]:
            source = self.root / record["path"]
            entry = cache / str(len(cached))
            entry.write_bytes(source.read_bytes())
            cached[(record["repo"], record["revision"], Path(record["path"]).name)] = entry
        (self.root / pins["files"][0]["path"]).unlink()
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = lambda repo_id, revision, filename: cached[(repo_id, revision, filename)]
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            import download_assets
            with patch.object(verify_assets, "PINS", pins_path), patch.object(download_assets, "PINS", pins_path), patch.object(download_assets, "hf_hub_download", hub.hf_hub_download):
                bootstrap.ensure_assets(self.root)
                self.assertEqual(len(verify_assets.verified_pins(self.root)["files"]), 12)
        self.assertFalse(list(self.root.glob("**/tmp*")))

    def test_custom_default_voice_fails_before_conversion_or_installation(self):
        pins_path, _ = self.pins()
        voices = self.root / "voices"
        voices.mkdir()
        custom = voices / "default.safetensors"
        custom.write_bytes(b"user voice")
        with patch.object(verify_assets, "PINS", pins_path):
            with self.assertRaisesRegex(RuntimeError, "move it aside"):
                bootstrap.prepare(self.root, self.root)
        self.assertEqual(custom.read_bytes(), b"user voice")
        self.assertFalse((self.root / "models").exists())

    def test_install_is_idempotent_and_preserves_symlink_directory_and_unrelated_files(self):
        source = self.source()
        models = self.root / "cache with spaces"
        models.mkdir()
        linked = self.root / "models"
        linked.symlink_to(models, target_is_directory=True)
        unrelated = models / "old-model"
        unrelated.write_bytes(b"keep")
        target = linked / "model.kpack"
        bootstrap.install(source, target)
        before = target.stat()
        bootstrap.install(source, target)
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertEqual(target.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertTrue(linked.is_symlink())
        self.assertEqual(unrelated.read_bytes(), b"keep")

    def test_conflicting_file_or_dangling_symlink_is_never_replaced(self):
        source = self.source()
        target = self.source("custom voice", b"custom")
        with self.assertRaisesRegex(RuntimeError, "move it aside"):
            bootstrap.install(source, target)
        self.assertEqual(target.read_bytes(), b"custom")
        target.unlink()
        target.symlink_to(self.root / "missing")
        with self.assertRaises(RuntimeError):
            bootstrap.install(source, target)
        self.assertTrue(target.is_symlink())

    def test_partial_copy_failure_removes_temporary_and_never_publishes(self):
        source = self.source()
        target = self.root / "voices" / "default.safetensors"
        def fail(input_file, output):
            output.write(b"partial")
            raise OSError("disk full")
        with patch.object(bootstrap.shutil, "copyfileobj", fail):
            with self.assertRaisesRegex(OSError, "disk full"):
                bootstrap.install(source, target)
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
