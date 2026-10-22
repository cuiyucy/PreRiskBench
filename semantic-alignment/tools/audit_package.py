"""Metadata-only static release scan; does not print matched secret values."""
import ast
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
findings=[]
counts={'python':0,'files':0,'csv':0}
ip=re.compile(r'\b(?:10\.(?:\d+\.){2}\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)\b')
for p in sorted(ROOT.rglob('*')):
    if not p.is_file(): continue
    rel=p.relative_to(ROOT).as_posix()
    counts['files']+=1
    if p.is_symlink(): findings.append({'file':rel,'issue':'symlink'}); continue
    if p.suffix.lower() in {'.csv','.parquet','.pkl','.pickle','.npy','.npz','.pt','.pth','.faiss','.sqlite','.sqlite3','.db','.jsonl','.h5','.hdf5','.bin','.safetensors'}:
        findings.append({'file':rel,'issue':'excluded-data-or-model-extension'})
        if p.suffix=='.csv': counts['csv']+=1
    if any(part in {'__pycache__','.git','.venv','node_modules','dataset','benchmark','models'} for part in p.relative_to(ROOT).parts[:-1]):
        # Code namespaces named benchmark/models are permitted, not runtime data.
        if not (rel.startswith('pipelines/') and p.suffix in {'.py','.yaml','.json'}):
            findings.append({'file':rel,'issue':'runtime-directory'})
    if p.suffix not in {'.py','.yaml','.yml','.json','.md','.txt','.cff','.toml','.sha256'}: continue
    text=p.read_text(encoding='utf-8',errors='replace')
    if p.name not in {'audit_package.py','validation.json'}:
        if ip.search(text): findings.append({'file':rel,'issue':'private-endpoint'})
        if re.search(r'/home/[^\s]+|\bsk-[A-Za-z0-9_-]{16,}',text): findings.append({'file':rel,'issue':'private-path-or-key'})
    if p.suffix=='.py':
        counts['python']+=1
        try:
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,ast.Dict):
                    for key,value in zip(node.keys,node.values):
                        if isinstance(key,ast.Constant) and isinstance(key.value,str) and re.search(r'(?i)password|passwd|api_key|access_token',key.value):
                            if isinstance(value,ast.Constant) and isinstance(value.value,str) and value.value and '${' not in value.value:
                                findings.append({'file':rel,'issue':'literal-credential-dict','line':node.lineno})
        except SyntaxError as e: findings.append({'file':rel,'issue':'syntax','line':e.lineno})
print(json.dumps({'counts':counts,'findings':findings,'note':'Static scan is not a legal, ownership, or privacy clearance.'},indent=2))
sys.exit(bool(findings))
