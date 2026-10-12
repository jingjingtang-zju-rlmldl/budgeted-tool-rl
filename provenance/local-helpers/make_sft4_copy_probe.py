from pathlib import Path
root=Path(__file__).resolve().parents[1]/'outputs/budgeted-tool-rl/toolrl'
text=(root/'sft4_run.py').read_text(encoding='utf-8')
text=text.replace('read,rows,write,validate,checkpoint,sha256,CONDITIONS','read,rows,write,checkpoint,sha256,CONDITIONS\nfrom .sft4_copy_rules import augmented_validate as validate,amendment')
text=text.replace("choices=['collect','reserve','dev','test','recovery','signal']", "choices=['copy_probe']")
text=text.replace("choices=['base','old','new']", "choices=['old']")
text=text.replace("Path(plan['root'])/'eval'", "Path(plan['root'])/'copy-probe/eval'")
text=text.replace("source_sha256=plan['source_sha256'],**plan['eval_fixed']", "source_sha256=plan['source_sha256'],copy_probe_source_sha256=sha256(Path(__file__)),\n                copy_probe_amendment_sha256=sha256(Path(plan['root'])/'copy-probe-amendment.json'),**plan['eval_fixed']")
(root/'sft4_copy_probe.py').write_text(text,encoding='utf-8',newline='\n')
import py_compile
py_compile.compile(str(root/'sft4_copy_probe.py'),doraise=True)
py_compile.compile(str(root/'sft4_copy_rules.py'),doraise=True)
print('Conditional training probe sources compile; frozen sources untouched.')
