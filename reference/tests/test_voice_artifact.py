import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assets import AM_ADAM_SOURCE_SHA256, HF_REVISION
from voice_artifact import (
    HEADER,
    PAYLOAD_SIZE,
    VoiceArtifactError,
    read_voice,
    serialize_voice,
    write_voice,
)


class VoiceArtifactTests(unittest.TestCase):
    def test_round_trip_records_pinned_provenance_and_payload(self):
        data = np.zeros((510, 1, 256), dtype="<f4").tobytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "am_adam.kvoice"
            write_voice(path, "f32", (510, 1, 256), data)
            raw = path.read_bytes()

        fields = HEADER.unpack_from(raw)
        self.assertEqual(len(raw) - HEADER.size, PAYLOAD_SIZE)
        self.assertEqual(fields[5], HF_REVISION.encode("ascii"))
        self.assertEqual(fields[6], AM_ADAM_SOURCE_SHA256.encode("ascii"))
        self.assertEqual(fields[7].decode("ascii"), hashlib.sha256(data).hexdigest())

    def test_writer_rejects_wrong_shape_and_source_pin(self):
        data = b"\0" * PAYLOAD_SIZE
        with self.assertRaisesRegex(VoiceArtifactError, "shape"):
            serialize_voice("f32", (509, 1, 256), data)
        with self.assertRaisesRegex(VoiceArtifactError, "SHA-256"):
            serialize_voice("f32", (510, 1, 256), data, "0" * 64)

    def test_reader_rejects_payload_changes(self):
        data = np.zeros((510, 1, 256), dtype="<f4").tobytes()
        raw = bytearray(serialize_voice("f32", (510, 1, 256), data))
        raw[HEADER.size] ^= 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.kvoice"
            path.write_bytes(raw)
            with self.assertRaisesRegex(VoiceArtifactError, "payload SHA-256"):
                read_voice(path)


if __name__ == "__main__":
    unittest.main()
