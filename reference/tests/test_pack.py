import struct
import tempfile
import unittest
from pathlib import Path

from assets import ARTIFACTS
from pack import PackError, read_pack, write_pack


METADATA = {
    "format": "kokoro-native-tensor-pack",
    "format_version": 1,
    "model": "hexgrad/Kokoro-82M",
    "model_revision": "f3ff3571791e39611d31c381e3a41a3af07b4987",
    "source_revision": "dfb907a02bba8152ca444717ca5d78747ccb4bec",
    "voice": "af_heart",
    "checkpoint_sha256": ARTIFACTS["kokoro-v1_0.pth"][1],
    "config_sha256": ARTIFACTS["config.json"][1],
    "voice_sha256": ARTIFACTS["voices/af_heart.pt"][1],
}


def complete_model_tensors():
    return {
        "bert.embeddings.word_embeddings.weight": ("f32", (1,), struct.pack("<f", 1.0)),
        "bert_encoder.projection.weight": ("f32", (1,), struct.pack("<f", 1.0)),
        "predictor.projection.weight": ("f32", (1,), struct.pack("<f", 1.0)),
        "text_encoder.projection.weight": ("f32", (1,), struct.pack("<f", 1.0)),
        "decoder.projection.weight": ("f32", (1,), struct.pack("<f", 1.0)),
        "voice.af_heart.style": ("f32", (1, 1, 256), struct.pack("<256f", *([0.25] * 256))),
    }


class TensorPackTests(unittest.TestCase):
    def test_round_trip_keeps_names_shapes_and_float32_values(self):
        tensors = complete_model_tensors()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.kokoro"
            write_pack(path, tensors, METADATA)
            pack = read_pack(path)

        self.assertEqual(set(pack.tensors), set(tensors))
        self.assertEqual(pack.tensors["bert.embeddings.word_embeddings.weight"].shape, (1,))
        self.assertEqual(
            pack.tensors["bert.embeddings.word_embeddings.weight"].data,
            struct.pack("<f", 1.0),
        )
        self.assertEqual(pack.metadata["voice"], "af_heart")

    def test_writer_aligns_payload_and_tensor_data(self):
        tensors = complete_model_tensors()
        tensors["predictor.aaa"] = ("i64", (1,), struct.pack("<q", 7))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aligned.kokoro"
            write_pack(path, tensors, METADATA)
            raw = path.read_bytes()
            _, _, _, index_size, metadata_size, _, _ = struct.unpack_from("<8sIIIIQQ", raw)
            payload_start = 40 + index_size + metadata_size
            pack = read_pack(path)

        self.assertEqual(payload_start % 8, 0)
        self.assertEqual(pack.tensors["predictor.aaa"].data, struct.pack("<q", 7))

    def test_reader_rejects_misaligned_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "misaligned.kokoro"
            write_pack(path, complete_model_tensors(), METADATA)
            raw = bytearray(path.read_bytes())
            index_size = struct.unpack_from("<I", raw, 16)[0]
            metadata_size = struct.unpack_from("<I", raw, 20)[0]
            payload_start = 40 + index_size + metadata_size
            raw[payload_start:payload_start] = b" "
            struct.pack_into("<I", raw, 20, metadata_size + 1)
            path.write_bytes(raw)

            with self.assertRaisesRegex(PackError, "8-byte aligned"):
                read_pack(path)

    def test_rejects_truncated_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.kokoro"
            path.write_bytes(b"KOKORO1\0\x01")

            with self.assertRaisesRegex(PackError, "truncated"):
                read_pack(path)

    def test_rejects_incompatible_format_version(self):
        metadata = {**METADATA, "format_version": 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.kokoro"
            write_pack(path, {}, metadata)

            with self.assertRaisesRegex(PackError, "version"):
                read_pack(path)

    def test_rejects_duplicate_tensor_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.kokoro"
            write_pack(path, {"voice.af_heart.style": ("f32", (1, 1, 256), b"\0" * 1024)}, METADATA)
            raw = bytearray(path.read_bytes())
            index_size = struct.unpack_from("<I", raw, 16)[0]
            index = raw[40 : 40 + index_size]
            metadata_start = 40 + index_size
            metadata_size = struct.unpack_from("<I", raw, 20)[0]
            payload_start = metadata_start + metadata_size
            metadata = bytes(raw[metadata_start:payload_start]).rstrip(b" ")
            metadata = metadata.replace(b'"tensor_count":1', b'"tensor_count":2')
            metadata += b" " * (-(40 + index_size * 2 + len(metadata)) % 8)
            payload = raw[payload_start:]
            raw = bytearray(
                struct.pack(
                    "<8sIIIIQQ",
                    b"KOKORO1\0",
                    1,
                    2,
                    index_size * 2,
                    len(metadata),
                    len(payload),
                    0,
                )
                + index
                + index
                + metadata
                + payload
            )
            path.write_bytes(raw)

            with self.assertRaisesRegex(PackError, "duplicate"):
                read_pack(path)

    def test_rejects_invalid_tensor_name_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-name.kokoro"
            tensors = {
                "bert.ok": ("f32", (1,), b"\0" * 4),
                "voice.af_heart.style": ("f32", (1, 1, 256), b"\0" * 1024),
            }
            write_pack(path, tensors, METADATA)
            raw = bytearray(path.read_bytes())
            _, _, rank, _ = struct.unpack_from("<HBBI", raw, 40)
            name_start = 40 + 8 + rank * 8 + 16
            raw[name_start] = ord("X")
            path.write_bytes(raw)

            with self.assertRaisesRegex(PackError, "tensor name"):
                read_pack(path)

            path = Path(directory) / "bad-meta.kokoro"
            write_pack(path, {"voice.af_heart.style": ("f32", (1,), b"\0\0\0\0")}, {"format": "other"})
            with self.assertRaisesRegex(PackError, "metadata"):
                read_pack(path)

    def test_rejects_model_revision_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-model.kokoro"
            metadata = {**METADATA, "model_revision": "d" * 40}
            tensors = {
                "bert.ok": ("f32", (1,), b"\0" * 4),
                "voice.af_heart.style": ("f32", (1, 1, 256), b"\0" * 1024),
            }
            write_pack(path, tensors, metadata)

            with self.assertRaisesRegex(PackError, "model revision"):
                read_pack(path)

    def test_rejects_artifact_digest_mismatch(self):
        metadata = {**METADATA, "config_sha256": "d" * 64}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-artifact.kokoro"
            write_pack(path, complete_model_tensors(), metadata)

            with self.assertRaisesRegex(PackError, "pinned config_sha256"):
                read_pack(path)

    def test_rejects_missing_model_tensor_groups(self):
        tensors = {
            "bert.not_a_real_model_parameter": ("f32", (1,), struct.pack("<f", 1.0)),
            "voice.af_heart.style": ("f32", (1, 1, 256), struct.pack("<256f", *([0.25] * 256))),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete-model.kokoro"
            write_pack(path, tensors, METADATA)

            with self.assertRaisesRegex(PackError, "missing required model tensor groups.*bert_encoder"):
                read_pack(path)

    def test_rejects_voice_tensor_with_incompatible_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-voice-shape.kokoro"
            tensors = {
                "bert.ok": ("f32", (1,), b"\0" * 4),
                "voice.af_heart.style": ("f32", (1, 1, 255), b"\0" * 1020),
            }
            write_pack(path, tensors, METADATA)

            with self.assertRaisesRegex(PackError, "voice tensor shape"):
                read_pack(path)

    def test_rejects_tensor_data_outside_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outside.kokoro"
            write_pack(path, {"voice.af_heart.style": ("f32", (1, 1, 256), b"\0" * 1024)}, METADATA)
            raw = bytearray(path.read_bytes())
            index_start = 40
            name_length, _, rank, _ = struct.unpack_from("<HBBI", raw, index_start)
            range_offset = index_start + 8 + rank * 8
            struct.pack_into("<Q", raw, range_offset, 1_000_000)
            path.write_bytes(raw)

            with self.assertRaisesRegex(PackError, "outside payload"):
                read_pack(path)


if __name__ == "__main__":
    unittest.main()
