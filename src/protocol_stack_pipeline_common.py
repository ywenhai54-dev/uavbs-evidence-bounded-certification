from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


WORKSPACE = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = WORKSPACE / config_path
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw)
    return config, config_path, hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p).lower()):
        digest.update(str(path).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_flag(
    path: Path,
    stage: str,
    status: str,
    expected_cases: int,
    actual_cases: int,
    config_hash: str,
    input_hash: str,
) -> None:
    lines = [
        f"stage: {stage}",
        f"status: {status}",
        f"expected_cases: {expected_cases}",
        f"actual_cases: {actual_cases}",
        f"timestamp: {utc_timestamp()}",
        f"config_hash: {config_hash}",
        f"input_hash: {input_hash}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_flag(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result

