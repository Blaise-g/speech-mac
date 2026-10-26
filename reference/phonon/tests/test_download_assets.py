"""Pinned download setup must reject corruption and select exact revisions."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download_assets


class DownloadAssets(unittest.TestCase):
    def test_exact_pin_and_verified_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached = root / "cached"
            cached.write_bytes(b"pinned asset")
            pins = root / "pins.json"
            pins.write_text(json.dumps({"files": [{"repo": "owner/model", "revision": "abc123",
                "path": "reference-codec/model.gguf", "size_bytes": 12,
                "sha256": hashlib.sha256(b"pinned asset").hexdigest()}]}))
            with patch.object(download_assets, "PINS", pins), patch.object(
                    download_assets, "hf_hub_download", return_value=str(cached)) as fetch:
                self.assertEqual(download_assets.download(root / "assets"), 1)
                fetch.assert_called_once_with(repo_id="owner/model", revision="abc123", filename="model.gguf")
                self.assertEqual((root / "assets/reference-codec/model.gguf").read_bytes(), b"pinned asset")
                cached.write_bytes(b"corrupt data")
                with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                    download_assets.download(root / "assets")
                self.assertEqual((root / "assets/reference-codec/model.gguf").read_bytes(), b"pinned asset")
