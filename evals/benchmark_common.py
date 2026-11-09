"""Shared identities, length buckets and deterministic benchmark sampling."""

from __future__ import annotations

import hashlib
from typing import Any

LENGTH_BUCKETS = (
    (1, 512, "short"),
    (513, 1024, "medium"),
    (1025, 2048, "long"),
    (2049, 4096, "extended"),
)
LENGTH_DEFINITION = "non-padding model input tokens, including SOS and EOS"
ACCURACY_DEFINITION = (
    "target-aligned tokens excluding SOS and padding; missing predictions are incorrect"
)


def length_bucket(length: int) -> str:
    for low, high, name in LENGTH_BUCKETS:
        if low <= length <= high:
            return name
    return "empty" if length <= 0 else "over_4096"


def length_bucket_range(length: int) -> str:
    for low, high, _ in LENGTH_BUCKETS:
        if low <= length <= high:
            return f"{low}_{high}"
    return length_bucket(length)


def normalize_bucket(bucket: str) -> str:
    """Read legacy numeric bucket names without substituting global scores."""
    for low, high, name in LENGTH_BUCKETS:
        if bucket in {name, f"{low}_{high}"}:
            return name
    if bucket in {"empty", "over_4096"}:
        return bucket
    raise ValueError(f"Unknown benchmark bucket: {bucket!r}")


def sample_metadata(batch: dict[str, Any], row: int) -> dict[str, Any]:
    ids = batch.get("sample_ids")
    files = batch.get("source_files")
    indices = batch.get("source_indices")
    if ids is None:
        raise ValueError("Benchmark samples require stable sample_ids")
    result = {"sample_id": str(ids[row])}
    if files is not None:
        result["source_file"] = str(files[row])
    if indices is not None:
        result["source_index"] = int(indices[row])
    return result


def records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record["sample_id"])
        if key in result:
            raise ValueError(f"Duplicate benchmark sample_id: {key}")
        result[key] = record
    return result


def sample_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def select_comparison_samples(
    records: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Round-robin populated buckets; hash rank makes selection order invariant."""
    if count < 0:
        raise ValueError("num_plots must be nonnegative")
    records_by_id(records)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        bucket = normalize_bucket(record["length_bucket"])
        groups.setdefault(bucket, []).append(record)
    order = [name for _, _, name in LENGTH_BUCKETS] + ["over_4096", "empty"]
    for group in groups.values():
        group.sort(key=lambda r: sample_rank(str(r["sample_id"]), seed))
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < min(count, len(records)):
        for bucket in order:
            group = groups.get(bucket, [])
            if cursor < len(group) and len(selected) < count:
                record = group[cursor]
                selected.append(
                    {
                        "sample_id": str(record["sample_id"]),
                        "length_bucket": bucket,
                        "target_length": int(record["target_length"]),
                    }
                )
        cursor += 1
    return selected
