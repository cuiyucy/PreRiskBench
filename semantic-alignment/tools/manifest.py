"""Create or verify public-file hashes; rejects symlinks and excludes itself only by name of manifest."""
import argparse
import hashlib
from pathlib import Path

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--verify',action='store_true')
args=parser.parse_args()
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=root/'MANIFEST.sha256'
if args.verify:
    expected={line.split('  ',1)[1]:line.split('  ',1)[0] for line in manifest.read_text(encoding='utf-8').splitlines()}
    current={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    if current!=expected: raise SystemExit('Manifest mismatch')
    print('Manifest verified:',len(current),'files')
else:
    files=sorted(p for p in root.rglob('*') if p.is_file() and p!=manifest)
    if any(p.is_symlink() for p in files): raise SystemExit('Refusing symlink')
    manifest.write_text(''.join(digest(p)+'  '+p.relative_to(root).as_posix()+'\n' for p in files))
