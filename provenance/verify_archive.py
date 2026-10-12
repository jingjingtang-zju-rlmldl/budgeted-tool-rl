"""Verify archived file bytes without loading models or running experiments."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
inventory = json.loads((root/'provenance/archive-inventory.json').read_text(encoding='utf-8'))
for name, record in inventory['files'].items():
    path = root/name
    assert path.is_file(), f'Missing archived file: {name}'
    assert path.stat().st_size == record['bytes'], f'Size differs: {name}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], f'Bytes differ: {name}'
print(f"Verified {len(inventory['files'])} archived files; no inference or training performed.")
