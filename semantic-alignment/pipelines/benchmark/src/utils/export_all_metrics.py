"""Export all task metrics to all_tasks_metrics.txt"""
import os as _release_os
from release_runtime import expand_path as _release_expand, expand_config as _release_expand_config
import json
from pathlib import Path
from collections import defaultdict
output_dir = Path(_release_expand('${BENCHMARK_WORK_DIR}/outputs'))
lines = []
for task_dir in sorted(output_dir.iterdir()):
    if not task_dir.is_dir() or task_dir.name.startswith('experiment') or task_dir.name == 'analysis':
        continue
    task = task_dir.name
    lines.append(f"{'=' * 100}")
    lines.append(f' {task}')
    lines.append(f"{'=' * 100}")
    data = {}
    for p in sorted(task_dir.rglob('metrics.json')):
        d = json.load(open(p))
        parts = p.relative_to(task_dir).parts
        exp, ds, algo = (parts[0], parts[1], parts[2])
        data[exp, ds, algo] = d
    exps = sorted(set((k[0] for k in data)))
    for exp in exps:
        dirs = sorted(set((k[1] for k in data if k[0] == exp)))
        for ds in dirs:
            lines.append(f'\n  {exp} | {ds}')
            lines.append(f"  {'算法':24s} {'Macro-F1':>9s} {'Accuracy':>9s} {'QWK':>9s} {'Kappa':>9s} {'MAE':>9s} {'RMSE':>9s} {'n_samples':>10s}")
            lines.append(f"  {'-' * 90}")
            for algo in ['lr', 'rf', 'xgboost']:
                an = {'lr': 'Logistic Regression', 'rf': 'Random Forest', 'xgboost': 'XGBoost'}[algo]
                d = data.get((exp, ds, algo))
                if d:
                    mf1 = d.get('Macro-F1', '')
                    acc = d.get('Accuracy', '')
                    qwk = d.get('QWK', '')
                    kap = d.get('Kappa', '')
                    mae = d.get('MAE', '')
                    rmse = d.get('RMSE', '')
                    n = d.get('n_samples', '')
                    try:
                        lines.append(f'  {an:24s} {float(mf1):>9.4f} {float(acc):>9.4f} {float(qwk):>9.4f} {float(kap):>9.4f} {float(mae):>9.4f} {float(rmse):>9.4f} {str(n):>10}')
                    except:
                        lines.append(f'  {an:24s} {str(mf1):>9} {str(acc):>9} {str(qwk):>9} {str(kap):>9} {str(mae):>9} {str(rmse):>9} {str(n):>10}')
out_path = output_dir / 'all_tasks_metrics.txt'
with open(out_path, 'w') as f:
    f.write('\n'.join(lines))
import os
fsize = os.path.getsize(out_path)
n_lines = len(lines)
n_records = sum((1 for task_dir in output_dir.iterdir() if task_dir.is_dir() for _ in task_dir.rglob('metrics.json')))
print(f'Written {n_lines} lines, {fsize / 1024:.0f} KB to {out_path}')
print(f'Total metrics records: {n_records}')
