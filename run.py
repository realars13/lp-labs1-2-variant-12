"""Самостоятельный запуск лабораторных данного комплекта."""
import argparse
import platform
import importlib.metadata
from common import ROOT, save_json
from generate_data import generate

LABS = ['1', '2']

def main():
    parser=argparse.ArgumentParser(description='Мягкий Арсений Сергеевич, группа 517')
    parser.add_argument('--lab',choices=LABS)
    parser.add_argument('--regenerate',action='store_true')
    args=parser.parse_args()
    if args.regenerate or not all((ROOT/f'lab{k}/data.json').exists() for k in LABS):
        generate()
    versions={p:importlib.metadata.version(p) for p in ['numpy', 'highspy', 'matplotlib', 'networkx']}
    save_json(ROOT/'environment.json',{'python':platform.python_version(),'platform':platform.system(),'packages':versions})
    import reports
    for k in ([args.lab] if args.lab else LABS):
        module=__import__(f'lab{k}.run',fromlist=['run'])
        result=module.run()
        getattr(reports,f'report{k}')(result)
        print(f'ЛР {k}: расчёт, независимые проверки и отчёт готовы.')

if __name__=='__main__':main()
