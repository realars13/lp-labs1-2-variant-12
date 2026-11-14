from copy import deepcopy
import numpy as np
from common import *

def build_model(d):
    return matrix_model(-np.array(d['priority']), d['A'], [-np.inf]*4, d['capacities'], d['demand'])

def verify(d, x):
    x = np.array(x); used=np.array(d['A'])@x
    err=max(0.,float(max(-x)),float(max(x-np.array(d['demand']))),float(max(used-np.array(d['capacities']))))
    return {'ok':err<=TOL,'max_violation':err,'objective':-float(np.array(d['priority'])@x),
            'resources_used':used,'resources_slack':np.array(d['capacities'])-used,
            'lower_slack':x,'upper_slack':np.array(d['demand'])-x}

def benchmark():
    out=ROOT/'lab1/results'; rng=np.random.default_rng(SEED+100)
    records=[]; dataset=[]
    for k in range(1,11):
        n=30*k; m=15*k
        A=rng.uniform(.05,1.,(m,n)); A*=rng.random((m,n))<.25
        upper=rng.uniform(5,15,n); b=(A@upper)*rng.uniform(.25,.55,m)
        c=-rng.uniform(1,10,n); model=matrix_model(c,A,[-np.inf]*m,b,upper)
        dataset.append({'n':n,'m':m,'A':A,'b':b,'c':c,'upper':upper})
        previous=None
        for alg in ['simplex','ipm']:
            samples=[]; result=None; errors=[]; objective_errors=[]
            for rep in range(3):
                result=solve(model,out/f'logs/benchmark_{k}_{alg}_{rep}.log',algorithm=alg)
                assert result['status']=='Optimal'
                check=finish_check(matrix_check(model,result['x']),result['objective'])
                errors.append(check['max_violation']);objective_errors.append(check['objective_error'])
                samples.append(result['seconds'])
            if previous is not None: assert abs(previous-result['objective'])<=TOL
            previous=result['objective']
            records.append({'n':n,'m':m,'algorithm':alg,'seconds_median':float(np.median(samples)),
                            'objective':result['objective'],'simplex_iterations':result['simplex_iterations'],
                            'ipm_iterations':result['ipm_iterations'],'crossover_iterations':result['crossover_iterations'],
                            'max_violation':max(errors),'max_objective_error':max(objective_errors)})
    save_json(ROOT/'lab1/benchmark_data.json',dataset); csv_write(out/'benchmark.csv',records)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for alg in ['simplex','ipm']:
        r=[a for a in records if a['algorithm']==alg]
        axes[0].plot([a['n'] for a in r],[a['seconds_median']*1000 for a in r],'-o',label=alg)
        axes[1].plot([a['n'] for a in r],[a[alg+'_iterations'] for a in r],'-o',label=alg)
    for ax in axes: ax.set_xlabel('Число переменных'); ax.grid(alpha=.3); ax.legend()
    axes[0].set_ylabel('Медиана времени решения, мс'); axes[1].set_ylabel('Итерации основного алгоритма')
    figure_save(out/'benchmark.png')
    return records

def run():
    d=load_data('lab1'); out=ROOT/'lab1/results'; out.mkdir(exist_ok=True)
    base=solve(build_model(d),out/'logs/base.log'); assert base['status']=='Optimal'
    shadow=-np.array(base['row_dual']); key=int(np.argmax(shadow))
    deficit=deepcopy(d); deficit['capacities'][key]*=.8
    stress=deepcopy(deficit); stress['capacities'][key]*=.75
    other=(key+1)%4; stress['capacities'][other]*=.6
    runs={}; scenarios=[]; checks={}; resource_rows=[]; solution=[]
    for name, data in [('base',d),('deficit',deficit),('stress',stress)]:
        r=base if name=='base' else solve(build_model(data),out/f'logs/{name}.log')
        assert r['status']=='Optimal'; ch=finish_check(verify(data,r['x']),r['objective'])
        check=matrix_check(build_model(data),r['x']); assert check['ok']
        checks[name]=ch; runs[name]=r
        actual=-r['objective']; prediction=-base['objective']+shadow@(np.array(data['capacities'])-d['capacities'])
        scenarios.append({'scenario':name,'status':r['status'],'utility':actual,'shadow_prediction':prediction,
                          'prediction_error':actual-prediction,'seconds':r['seconds']})
        for j,x in enumerate(r['x']): solution.append({'scenario':name,'task':d['tasks'][j],'processed':x,'demand':d['demand'][j],'priority':d['priority'][j]})
        for i,res in enumerate(d['resources']): resource_rows.append({'scenario':name,'resource':res,'capacity':data['capacities'][i],
               'used':ch['resources_used'][i],'slack':ch['resources_slack'][i],'shadow_price':-r['row_dual'][i]})
    sensitivity=[]
    for i in range(4):
        pert=deepcopy(d); pert['capacities'][i]+=.01
        r=solve(build_model(pert),out/f'logs/shadow_{i}.log')
        finish_check(verify(pert,r['x']),r['objective'])
        sensitivity.append({'resource':d['resources'][i],'delta':.01,'predicted_gain':shadow[i]*.01,'actual_gain':base['objective']-r['objective']})
    csv_write(out/'solution.csv',solution);csv_write(out/'resources.csv',resource_rows)
    csv_write(out/'scenarios.csv',scenarios);csv_write(out/'shadow_check.csv',sensitivity)
    save_json(out/'verification.json',checks);save_json(out/'solver_results.json',runs)
    save_json(out/'scenario_data.json',{'base':d,'deficit':deficit,'stress':stress})
    # Отрицательные контрольные примеры: нарушения ресурса и нижней границы.
    assert not verify(d,np.array(d['demand'])*2)['ok']
    bad=np.array(base['x']); bad[0]=-1; assert not verify(d,bad)['ok']
    fig,ax=plt.subplots(figsize=(8,4)); xx=np.arange(6)
    for k,name in enumerate(runs):ax.bar(xx+(k-1)*.25,runs[name]['x'],.25,label=name)
    ax.set_xticks(xx,d['tasks']);ax.set_ylabel('Обработано, тыс. задач');ax.legend();figure_save(out/'plans.png')
    bench=benchmark()
    return {'data':d,'scenarios':scenarios,'resources':resource_rows,'sensitivity':sensitivity,'runs':runs,'checks':checks,'key':key,'other':other,'benchmark':bench}
