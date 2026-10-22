"""Verify, without exposing content, an authorized local cluster dictionary."""
import argparse
import hashlib
from pathlib import Path

EXPECTED='744e95a840ede013b0d9cf96838327c21d0b1ebc3ceadee44c85ac3ea582368c'
parser=argparse.ArgumentParser()
parser.add_argument('artifact',type=Path)
args=parser.parse_args()
if not args.artifact.is_file(): raise SystemExit('Artifact not found')
actual=hashlib.sha256(args.artifact.read_bytes()).hexdigest()
if actual!=EXPECTED: raise SystemExit('Hash mismatch: expected '+EXPECTED+', got '+actual)
print('Frozen paper cluster artifact verified:',actual)
