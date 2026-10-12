"""Download a fixed official model snapshot into a caller-selected directory."""
import argparse
import json
import os
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path, default=Path('models/model-location.json'))
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(MODEL_ID, revision=REVISION, cache_dir=str(args.cache.resolve()), max_workers=2,
                                 allow_patterns=['*.json', '*.safetensors', 'tokenizer*', 'merges.txt', 'vocab.json', '*.jinja'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(model_id=MODEL_ID, revision=REVISION, snapshot=snapshot), indent=2), encoding='utf-8')
    print('Pinned model available at', snapshot)


if __name__ == '__main__':
    main()
