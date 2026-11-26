"""Pre-tokenized data must not acquire an unrelated tokenizer identity."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from terminus.integrity import file_sha256
from terminus.pack_tokens import pack_token_streams


TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer/tokenizer-65k.json"


@pytest.fixture
def packed(tmp_path):
    for split in ("train", "val"):
        shard = tmp_path / f"en.{split}.00000.bin"
        np.asarray([99, 75, 75, 570, 4] * 4, dtype=np.uint16).tofile(shard)
        shard.with_suffix(".json").write_text(json.dumps({
            "tokenizer_sha256": file_sha256(TOKENIZER), "tokens": 20,
            "documents": 4, "utf8_bytes": 40,
        }))
    return tmp_path


def assemble(folder):
    return pack_token_streams(folder, TOKENIZER, source_mix={"en": 1.0},
                              total_train_tokens=8, seq_len=4, batch_size=1,
                              scope={"name": "test-en-v1", "languages": ["en"], "domains": []},
                              languages_by_stream={"en": "en"})


def test_valid_inventory_preserves_tokenizer_and_shard_identity(packed):
    manifest = assemble(packed)
    assert manifest.tokenizer["sha256"] == file_sha256(TOKENIZER)
    assert len(manifest.shards) == 2
    assert all(s.num_tokens == 20 and s.sha256 == file_sha256(packed / s.path)
               for s in manifest.shards)


@pytest.mark.parametrize("failure", ["missing_sidecar", "missing_identity", "wrong_identity", "wrong_count", "partial_token"])
def test_unproven_or_inconsistent_shards_cannot_be_relabelled(packed, failure):
    shard = packed / "en.train.00000.bin"
    sidecar = shard.with_suffix(".json")
    record = json.loads(sidecar.read_text())
    if failure == "missing_sidecar":
        sidecar.unlink()
    elif failure == "partial_token":
        with shard.open("ab") as handle:
            handle.write(b"\x00")
    else:
        if failure == "missing_identity":
            record.pop("tokenizer_sha256")
        elif failure == "wrong_identity":
            record["tokenizer_sha256"] = "0" * 64
        else:
            record["tokens"] -= 1
        sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        assemble(packed)
    assert not (packed / "manifest.json").exists()


def test_existing_tokenizer_is_checked_before_manifest_identity_is_written(packed):
    destination = packed / "tokenizer.json"
    shutil.copyfile(TOKENIZER, destination)
    with destination.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="existing packed tokenizer"):
        assemble(packed)
    assert not (packed / "manifest.json").exists()
