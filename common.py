"""Решение моделей HiGHS и сохранение результатов. Все цели — минимум."""
import csv
import json
import re
from pathlib import Path
from time import perf_counter
import numpy as np
import highspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TOL = 1e-7
SEED = 10114
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

def save_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding='utf-8')

def load_data(lab):
    return json.loads((ROOT / lab / 'data.json').read_text(encoding='utf-8'))

def csv_write(path, rows):
    rows = list(rows)
    if not rows: return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def matrix_model(c, A, lo, hi, upper, integer=()):
    return {'c': np.asarray(c, float), 'A': np.asarray(A, float),
            'lo': np.asarray(lo, float), 'hi': np.asarray(hi, float),
            'upper': np.asarray(upper, float), 'integer': list(integer)}

def solve(model, log, algorithm='simplex', presolve=True, limit=60):
    """Новый экземпляр для каждого запуска; время только h.run, без сборки."""
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    # HiGHS дописывает лог; очистка обеспечивает воспроизводимый состав файла.
    log.write_text('', encoding='utf-8')
    h = highspy.Highs()
    options = {'log_to_console': False, 'log_file': str(log), 'threads': 1,
               'random_seed': SEED, 'presolve': 'on' if presolve else 'off',
               'primal_feasibility_tolerance': 1e-8,
               'dual_feasibility_tolerance': 1e-8,
               'mip_feasibility_tolerance': 1e-8, 'mip_rel_gap': 0.0,
               'mip_abs_gap': 0.0, 'time_limit': float(limit)}
    if not model['integer']: options['solver'] = algorithm
    for key, val in options.items():
        assert h.setOptionValue(key, val) == highspy.HighsStatus.kOk, key
    c, A = model['c'], model['A']
    n = len(c)
    h.addCols(n, c, np.zeros(n), model['upper'], 0, np.zeros(n+1, np.int32), np.array([],np.int32), np.array([],float))
    rows, cols = np.nonzero(A)
    starts = np.r_[0, np.cumsum(np.bincount(rows, minlength=A.shape[0]))].astype(np.int32)
    h.addRows(A.shape[0], model['lo'], model['hi'], len(cols), starts, cols.astype(np.int32), A[rows, cols])
    if model['integer']:
        idx = np.array(model['integer'], np.int32)
        h.changeColsIntegrality(len(idx), idx, np.full(len(idx), highspy.HighsVarType.kInteger, np.uint8))
    start = perf_counter(); h.run(); elapsed = perf_counter() - start
    status = h.modelStatusToString(h.getModelStatus())
    info, sol = h.getInfo(), h.getSolution()
    valid = info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible
    result = {'status': status, 'seconds': elapsed, 'objective': float(info.objective_function_value) if valid else None,
              'x': list(sol.col_value) if valid else None,
              'row_dual': list(sol.row_dual) if sol.dual_valid else None,
              'col_dual': list(sol.col_dual) if sol.dual_valid else None,
              'simplex_iterations': info.simplex_iteration_count,
              'ipm_iterations': info.ipm_iteration_count,
              'crossover_iterations': info.crossover_iteration_count,
              'nodes': int(info.mip_node_count) if model['integer'] else None,
              'mip_gap': float(info.mip_gap) if model['integer'] and valid and np.isfinite(info.mip_gap) else None,
              'dual_bound': float(info.mip_dual_bound) if model['integer'] and np.isfinite(info.mip_dual_bound) else None,
              'variables': n, 'constraints': A.shape[0]}
    h.clear()
    timing=re.search(r'^\s+([0-9.]+) \(Solve\)',log.read_text(encoding='utf-8'),re.MULTILINE)
    result['solve_phase_seconds']=float(timing.group(1)) if timing else None
    return result

def matrix_check(model, x):
    """Проверяет массивы, не используя объект или методы солвера."""
    x = np.asarray(x); v = model['A'] @ x
    violations = [0., float(np.max(-x)), float(np.max(x-model['upper'])),
                  float(np.max(model['lo']-v)), float(np.max(v-model['hi']))]
    if model['integer']:
        z = x[model['integer']]; violations.append(float(np.max(np.abs(z-np.round(z)))))
    error = max(violations)
    return {'ok': bool(error <= TOL), 'max_violation': error, 'objective': float(model['c'] @ x)}

def finish_check(check, solver_obj):
    check['objective_error'] = abs(check['objective']-solver_obj)
    check['ok'] = bool(check['ok'] and check['objective_error'] <= TOL)
    assert check['ok'], check
    return check

def table(rows, keys=None):
    if not rows: return 'Нет строк.'
    keys = keys or list(rows[0])
    def fmt(v):
        if isinstance(v, (float, np.floating)): return f'{v:.6g}'
        return '—' if v is None else str(v)
    return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|'] +
                     ['| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rows])

def write_report(lab, text, filename='REPORT.md'):
    (ROOT/lab/filename).write_text(text.strip()+'\n', encoding='utf-8')

def figure_save(path):
    plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches='tight'); plt.close()
