"""Replay tools and independently regrade recorded outputs without model inference."""
import argparse
import hashlib
import json
from pathlib import Path

from .environment import LookupEnvironment, parse_action, verify


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model-info',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    config=manifest['config']
    if manifest['status']!='complete':
        raise ValueError('run is incomplete')
    if hashlib.sha256(args.data.read_bytes()).hexdigest()!=config['dataset_sha256']:
        raise ValueError('dataset hash mismatch')
    tasks={t['task_id']:t for t in [json.loads(x) for x in args.data.read_text(encoding='utf-8').splitlines()]}
    rows=[json.loads(x) for x in (args.run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=config['task_count'] or len({r['task_id'] for r in rows})!=len(rows):
        raise ValueError('task accounting mismatch')
    info=json.loads(args.model_info.read_text(encoding='utf-8'))
    if info['revision']!=config['model_revision']:
        raise ValueError('model revision mismatch')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
    request_count=0
    for row in rows:
        task=tasks[row['task_id']]
        if row['expected']!=task['expected']:
            raise ValueError('expected answer changed')
        question_index=row.get('question_index',1)
        prefix=row['messages'][:question_index+1]
        if prefix[question_index]['role']!='user' or prefix[question_index]['content']!=task['prompt']:
            raise ValueError('question changed')
        env=LookupEnvironment(task,config['max_calls'])
        observations=[]
        final=None
        for request in row['requests']:
            text=tokenizer.apply_chat_template(prefix,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            expected_ids=tokenizer(text,add_special_tokens=False)['input_ids']
            if expected_ids!=request['input_ids']:
                raise ValueError('model input tokens do not match recorded messages')
            if request.get('context_limit'):
                break
            if tokenizer.decode(request['output_ids'],skip_special_tokens=True)!=request['text']:
                raise ValueError('decoded output does not match recorded token IDs')
            request_count+=1
            prefix.append({'role':'assistant','content':request['text']})
            if request['truncated']:
                break
            try:
                action=parse_action(request['text'],allow_fence=config.get('parser')=='fenced')
            except ValueError:
                break
            if action['action']=='answer':
                final=action['value']
                break
            try:
                observation=env.lookup(action['entity'],action['field'])
            except ValueError:
                break
            observations.append(observation)
            prefix.append({'role':'tool','content':json.dumps(observation,sort_keys=True)})
        computed=final is not None and verify(final,task['expected'])
        if computed!=row['success'] or observations!=row['observations'] or len(env.calls)!=row['calls']:
            raise ValueError('independent replay/grading mismatch')
    result=dict(tasks_verified=len(rows),requests_verified=request_count,tool_replay_matches=True,
                answer_regrading_matches=True,input_token_provenance_matches=True,output_token_decode_matches=True,
                episodes_sha256=hashlib.sha256((args.run/'episodes.jsonl').read_bytes()).hexdigest())
    (args.run/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
