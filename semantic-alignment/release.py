"""Safe publication entrypoint. Default commands never run real-data jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description='Perioperative semantic alignment research code')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('plan',help='Show reconstruction stages without reading data')
    sub.add_parser('check',help='Run synthetic tests and package audit only')
    bench=sub.add_parser('benchmark',help='ASA-only historical benchmark CLI')
    bench.add_argument('--execute',action='store_true',help='Explicitly permit local benchmark processing')
    bench.add_argument('args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.command=='plan':
        print(json.dumps({'scope':'ASA only','stages':[
            'Obtain independent source-data and terminology access',
            'Prepare authorized field-harmonized inputs for VitalDB, MOVER and INSPIRE',
            'Supply its authorized common feature and ASA-label outputs through the documented interface',
            'Extract text, exact match, retrieve/rerank, RAG assistance, merge mapping records',
            'Run frequency-first semantic clustering and verify the frozen paper artifact hash',
            'Quality filter and patient-grouped ASA train/validation/test construction',
            'Run six model families, six-direction retrieval/transfer, and paper-specific analyses'],
            'validation':'Synthetic interface tests only; no real-data or hosted-model rerun',
            'documentation':'See docs/RECONSTRUCTION.md for prerequisites and limitations'},indent=2))
    elif args.command=='check':
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        for command in [[sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],
                        [sys.executable,'-B','tools/audit_package.py']]:
            run=subprocess.run(command,cwd=ROOT,env=env)
            if run.returncode: raise SystemExit(run.returncode)
    else:
        rest=args.args
        if rest[:1]==['--']: rest=rest[1:]
        if not rest: rest=['--help']
        if rest[0] not in {'--help','build','run','status'}:
            parser.error('Supported benchmark commands: build, run, status')
        for i,arg in enumerate(rest):
            if arg in {'--task','-t'} and (i+1>=len(rest) or rest[i+1]!='asa'):
                parser.error('This publication entrypoint supports ASA only')
            if arg.startswith('--task=') and arg!='--task=asa': parser.error('ASA only')
        if rest[0] in {'build','run'} and not any(x in {'--task','-t','--task=asa'} for x in rest):
            rest.extend(['--task','asa'])
        if not args.execute and '--help' not in rest:
            print('PLAN ONLY:', ' '.join(['benchmark/main.py',*rest]))
            print('No files or database were read. See docs/RECONSTRUCTION.md before --execute.')
            return
        # All runtime defaults live under the public checkout, never historical host paths.
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT))
        benchroot=ROOT/'pipelines/benchmark'
        env.setdefault('BENCHMARK_WORK_DIR',str(benchroot))
        env.setdefault('ALIGNMENT_WORK_DIR',str(ROOT/'pipelines/alignment'))
        env.setdefault('LOCAL_WORK_DIR',str(ROOT/'runtime'))
        if rest[0]=='build' and '--dry-run' not in rest and '--help' not in rest:
            preflight=subprocess.run([sys.executable,'-B',str(ROOT/'tools/check_patient_maps.py')],cwd=benchroot,env=env)
            if preflight.returncode: raise SystemExit(preflight.returncode)
        raise SystemExit(subprocess.run([sys.executable,'-B',str(benchroot/'main.py'),*rest],cwd=benchroot,env=env).returncode)

if __name__=='__main__': main()
