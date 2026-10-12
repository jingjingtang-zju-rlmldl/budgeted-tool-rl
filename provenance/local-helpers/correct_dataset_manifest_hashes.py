"""Correct metadata only; preserve recorded JSONL bytes and audit the change."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent.parent/'outputs'/'budgeted-tool-rl'
audit=[]
for data in sorted((root/'data').glob('*.jsonl')):
    metadata=data.with_suffix('.manifest.json')
    old=json.loads(metadata.read_text(encoding='utf-8'))
    byte_hash=hashlib.sha256(data.read_bytes()).hexdigest()
    normalized_hash=hashlib.sha256(data.read_text(encoding='utf-8').encode()).hexdigest()
    assert old['sha256'] in (byte_hash,normalized_hash)
    if old['sha256']==byte_hash:
        continue
    new=dict(old,sha256=byte_hash,normalized_lf_sha256=normalized_hash,
             hash_correction='Corrected Windows newline byte accounting after inference; JSONL bytes unchanged.')
    metadata.write_text(json.dumps(new,indent=2)+'\n',encoding='utf-8')
    assert hashlib.sha256(data.read_bytes()).hexdigest()==byte_hash
    audit.append(dict(file=data.relative_to(root).as_posix(),before=old,after=new,data_bytes_unchanged=True))
(root/'results'/'prompt-control-v1'/'dataset-manifest-hash-correction.json').write_text(
    json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'corrected_metadata_files':len(audit),'task_files_modified':False}))
