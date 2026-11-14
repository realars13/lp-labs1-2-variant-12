from copy import deepcopy
from time import perf_counter
import networkx as nx
import numpy as np
from common import *

def build_model(d):
    nodes=d['nodes']; arcs=d['arcs']; sources=list(d['sources']); ne=len(arcs)
    B=np.zeros((len(nodes),ne+len(sources)))
    for e,a in enumerate(arcs):
        B[nodes.index(a['from']),e]=-1
        B[nodes.index(a['to']),e]=a['efficiency']
    for j,s in enumerate(sources): B[nodes.index(s),ne+j]=1
    rhs=[d['demand'].get(n,0) for n in nodes]
    return matrix_model([a['cost'] for a in arcs]+[0]*len(sources),B,rhs,rhs,
                        [a['capacity'] for a in arcs]+list(d['sources'].values()))

def verify(d,x):
    """Повторная сборка балансов непосредственно из дуг, без матрицы модели."""
    ne=len(d['arcs']); flows=np.array(x[:ne]); supply=np.array(x[ne:])
    net={n:0. for n in d['nodes']}
    for e,a in enumerate(d['arcs']):
        net[a['from']]-=flows[e]; net[a['to']]+=flows[e]*a['efficiency']
    for i,s in enumerate(d['sources']):net[s]+=supply[i]
    residual={n:net[n]-d['demand'].get(n,0) for n in net}
    arc_excess=flows-np.array([a['capacity'] for a in d['arcs']])
    source_excess=supply-np.array(list(d['sources'].values()))
    err=max(0.,max(abs(v) for v in residual.values()),float(max(arc_excess)),float(max(source_excess)),float(max(-np.array(x))))
    return {'ok':err<=TOL,'max_violation':err,'objective':sum(a['cost']*flows[e] for e,a in enumerate(d['arcs'])),
            'node_residuals':residual,'arc_slack':-arc_excess,'source_slack':-source_excess,
            'loss':sum((1-a['efficiency'])*flows[e] for e,a in enumerate(d['arcs'])),
            'injected':sum(supply),'delivered':sum(d['demand'].values())}

def greedy(d):
    """Дешёвые двухзвенные пути по стоимости одного доставленного м³."""
    start=perf_counter(); arcs=d['arcs']; sources=list(d['sources']); ne=len(arcs)
    paths=[]
    for e,a in enumerate(arcs):
        if a['from'] not in sources:continue
        for f,b in enumerate(arcs):
            if a['to']==b['from'] and b['to'] in d['demand']:
                ef=a['efficiency']*b['efficiency']
                price=a['cost']/ef+b['cost']/b['efficiency']
                paths.append((price,e,f,sources.index(a['from']),b['to']))
    flows=np.zeros(ne); supply=np.zeros(len(sources)); remaining=dict(d['demand'])
    for price,e,f,s,dest in sorted(paths):
        eta1=arcs[e]['efficiency'];eta2=arcs[f]['efficiency']
        amount=min(remaining[dest], (arcs[e]['capacity']-flows[e])*eta1*eta2,
                   (arcs[f]['capacity']-flows[f])*eta2,
                   (list(d['sources'].values())[s]-supply[s])*eta1*eta2)
        amount=max(0.,amount)
        flows[e]+=amount/(eta1*eta2);flows[f]+=amount/eta2;supply[s]+=amount/(eta1*eta2)
        remaining[dest]-=amount
    return np.r_[flows,supply],perf_counter()-start,remaining

def network_plot(d,r,out):
    # Два слоя одной сети разделены, чтобы подписи двадцати дуг читались.
    fig,axes=plt.subplots(1,2,figsize=(17,8))
    for part,ax in enumerate(axes):
        subset=[(e,a) for e,a in enumerate(d['arcs']) if (a['from'].startswith('S'))==(part==0)]
        left=[f'S{i+1}' for i in range(2)] if part==0 else [f'H{i+1}' for i in range(4)]
        right=[f'H{i+1}' for i in range(4)] if part==0 else [f'D{i+1}' for i in range(6)]
        pos={n:(0,i/(len(left)-1)) for i,n in enumerate(left)}
        pos.update({n:(1,i/(len(right)-1)) for i,n in enumerate(right)})
        g=nx.DiGraph();g.add_edges_from((a['from'],a['to']) for _,a in subset)
        nx.draw_networkx_nodes(g,pos,ax=ax,node_size=750,node_color='#dceaf1')
        nx.draw_networkx_labels(g,pos,ax=ax,font_size=10)
        nx.draw_networkx_edges(g,pos,ax=ax,arrows=True,node_size=750,alpha=.65)
        # Подписи разнесены по горизонтали в зависимости от начала дуги.
        for e,a in subset:
            t=.27+.14*(left.index(a['from'])%4)
            x1,y1=pos[a['from']];x2,y2=pos[a['to']]
            ax.text(x1+(x2-x1)*t,y1+(y2-y1)*t,f"{r['x'][e]:.1f}/{a['capacity']}",fontsize=8,
                    ha='center',va='center',bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':1})
        ax.set_xlim(-.16,1.16);ax.set_ylim(-.12,1.12);ax.axis('off')
        ax.set_title('Источники → узлы' if part==0 else 'Узлы → потребители')
    fig.suptitle('Базовый поток воды: отправлено / пропускная способность, м³/ч')
    figure_save(out/'network.png')

def run():
    d=load_data('lab2');out=ROOT/'lab2/results';out.mkdir(exist_ok=True)
    base=solve(build_model(d),out/'logs/base.log');assert base['status']=='Optimal'
    # Самая загруженная = наибольший абсолютный отправляемый объём; правило явно фиксируется.
    busiest=int(np.argmax(base['x'][:len(d['arcs'])]))
    growth=deepcopy(d);growth['demand']={k:v*1.2 for k,v in d['demand'].items()}
    failure=deepcopy(d);failure['arcs'][busiest]['capacity']=0
    overload=deepcopy(d);overload['arcs'][busiest]['capacity']*=.8
    scenarios=[];runs={};checks={};sol=[];nodes=[];sat=[];scenario_data={}
    for name,data in [('base',d),('deficit',growth),('stress',failure),('overload',overload)]:
        r=base if name=='base' else solve(build_model(data),out/f'logs/{name}.log')
        runs[name]=r;scenario_data[name]=data
        if r['x'] is not None:
            ch=finish_check(verify(data,r['x']),r['objective']);checks[name]=ch
            assert matrix_check(build_model(data),r['x'])['ok']
            for e,a in enumerate(data['arcs']):sol.append({'scenario':name,'arc':f"{a['from']}-{a['to']}",'sent':r['x'][e],
                 'received':r['x'][e]*a['efficiency'],'capacity':a['capacity'],'efficiency':a['efficiency'],'unit_cost':a['cost']})
            for n in data['nodes']:nodes.append({'scenario':name,'node':n,'demand':data['demand'].get(n,0),'balance_error':ch['node_residuals'][n]})
        else:
            # Не выдаём вектор недопустимого решения за найденный план.
            checks[name]={'ok':None,'reason':'Допустимый план отсутствует; проверка вектора не выполнялась.'}
        scenarios.append({'scenario':name,'status':r['status'],'cost':r['objective'],
                          'cost_change':r['objective']-base['objective'] if r['objective'] is not None else None,'seconds':r['seconds']})
    for e,a in enumerate(d['arcs']):
        sat.append({'arc':f"{a['from']}-{a['to']}",'flow':base['x'][e],'capacity':a['capacity'],
                    'saturated':abs(base['x'][e]-a['capacity'])<=TOL,
                    'capacity_value':max(0.,-base['col_dual'][e]) if base['x'][e]>TOL else 0.})
    xg,tg,unmet=greedy(d);cg=verify(d,xg)
    comparison={'feasible':cg['ok'],'greedy_cost':cg['objective'],'optimal_cost':base['objective'],
                'gap_percent':100*(cg['objective']-base['objective'])/base['objective'] if cg['ok'] else None,
                'greedy_seconds':tg,'solver_seconds':base['seconds'],'unmet_demand':unmet}
    save_json(out/'greedy.json',comparison);save_json(out/'greedy_verification.json',cg)
    csv_write(out/'greedy_solution.csv',[{'variable':i,'value':v} for i,v in enumerate(xg)])
    csv_write(out/'solution.csv',sol);csv_write(out/'node_balances.csv',nodes);csv_write(out/'scenarios.csv',scenarios)
    csv_write(out/'bottlenecks.csv',sat);save_json(out/'verification.json',checks)
    save_json(out/'solver_results.json',runs);save_json(out/'scenario_data.json',scenario_data)
    bad=np.array(base['x']);bad[0]+=1;assert not verify(d,bad)['ok']
    network_plot(d,base,out)
    return {'data':d,'runs':runs,'checks':checks,'scenarios':scenarios,'busiest':busiest,'bottlenecks':sat,'greedy':comparison}
