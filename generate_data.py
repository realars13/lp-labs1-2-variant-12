"""Генерация данных лабораторных работ данного комплекта."""
import numpy as np
from common import ROOT, SEED, save_json

def generate():
    rng = np.random.default_rng(SEED)
    A = rng.integers(1, 10, (4,6))
    demand = rng.integers(16, 31, 6)
    priority = rng.integers(5, 21, 6)
    capacities = np.floor((A@demand)*np.array([.43,.50,.47,.55]))
    save_json(ROOT/'lab1/data.json', {'seed':SEED,'tasks':[f'T{i+1}' for i in range(6)],
              'resources':['CPU','RAM','IO','NET'],'units':['ядро·ч','ГБ·ч','тыс. операций','ГБ'],
              'A':A,'demand':demand,'priority':priority,'capacities':capacities})
    rng = np.random.default_rng(SEED)
    arcs=[]
    for s in range(2):
        for h in range(4):
            arcs.append({'from':f'S{s+1}','to':f'H{h+1}','capacity':int(rng.integers(24,36)),
                         'cost':int(rng.integers(1,5)), 'efficiency':float(rng.choice([.94,.96,.98]))})
    for d in range(6):
        for h in [d%4,(d+1)%4]:
            arcs.append({'from':f'H{h+1}','to':f'D{d+1}','capacity':int(rng.integers(13,22)),
                         'cost':int(rng.integers(1,6)), 'efficiency':float(rng.choice([.90,.94,.96]))})
    nodes=[f'S{i+1}' for i in range(2)]+[f'H{i+1}' for i in range(4)]+[f'D{i+1}' for i in range(6)]
    save_json(ROOT/'lab2/data.json', {'seed':SEED,'nodes':nodes,'arcs':arcs,
              'sources':{'S1':65,'S2':65},'demand':{f'D{i+1}':int(rng.integers(13,18)) for i in range(6)}})

if __name__=='__main__': generate()
