import json
from pathlib import Path

root=Path(__file__).resolve().parent.parent/'outputs'/'budgeted-tool-rl'/'results'/'prompt-control-v1'
output=[]
for path in sorted(root.glob('*-seed*')):
    manifest=json.loads((path/'manifest.json').read_text())
    rows=[]
    if (path/'episodes.jsonl').exists():
        for line in (path/'episodes.jsonl').read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # An active writer may be between writing and flushing.
    output.append(dict(condition=path.name,status=manifest['status'],n=len(rows),
                       correct=sum(r['success'] for r in rows),strict=sum(r['strict_success'] for r in rows),
                       by_kind={k:[sum(r['success'] for r in rows if r['kind']==k),sum(r['kind']==k for r in rows)]
                                for k in ['direct','one_lookup','two_lookup']},
                       two_started=sum(r['calls']>0 for r in rows if r['kind']=='two_lookup')))
for item in output:
    print(json.dumps(item))
