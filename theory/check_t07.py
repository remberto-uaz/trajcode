"""T07: verificación numérica de la Parte 2 (desalineación estructural) y números del ejemplo."""
import random, math
from collections import defaultdict
exec(open('check_t06.py').read().split("L2=math.log2")[0])
L2=math.log2; LN2=math.log(2)
random.seed(21); bad=defaultdict(int); T=400
for t in range(T):
    succ,p,lab=rand_dag(random.randint(6,12),3,random.randint(2,4))
    q={v:dict(d) for v,d in p.items()}
    for v in q:
        if random.random()<0.5:
            w={u:x*random.uniform(0.2,4) for u,x in q[v].items()}; z=sum(w.values()); q[v]={u:x/z for u,x in w.items()}
    P=dict(walks(succ,p)); Q=dict(walks(succ,q)); mu={w:lab[w[-1]] for w in P}
    nu=defaultdict(float)
    for w,x in P.items():
        for v in w[:-1]: nu[v]+=x
    A=sum(x*L2(x/Q[w]) for w,x in P.items())
    Dv={v:sum(p[v][u]*L2(p[v][u]/q[v][u]) for u in p[v]) for v in p}
    tv={v:sum(abs(p[v][u]-q[v][u]) for u in p[v]) for v in p}
    lower=sum(nu[v]*tv[v]**2/(2*LN2) for v in p)
    upper=sum(nu[v]*L2(1/min(q[v].values())) for v in p if Dv[v]>1e-15)
    bad['(b) descomposicion']+= abs(A-sum(nu[v]*Dv[v] for v in p))>1e-9
    bad['(d) Pinsker inferior']+= A< lower-1e-12
    bad['(d) cota superior']+= A> upper+1e-12
    # (e) separacion significado / camino
    PM=defaultdict(float); QM=defaultdict(float)
    for w in P: PM[mu[w]]+=P[w]; QM[mu[w]]+=Q[w]
    Ds=sum(x*L2(x/QM[m]) for m,x in PM.items())
    bad['(e) separacion']+= abs(A-(Ds+sum(P[w]*L2((P[w]/PM[mu[w]])/(Q[w]/QM[mu[w]])) for w in P)))>1e-9
    bad['(e) Ds<=A']+= Ds>A+1e-12
    # Corolario receptor sin estadisticas: costo = sum nu(v) log2 outdeg(v)
    U={w:math.prod(1/len(succ[v]) for v in w[:-1]) for w in P}
    cost=sum(x*-L2(U[w]) for w,x in P.items())
    bad['corolario uniforme']+= abs(cost-sum(nu[v]*L2(len(succ[v])) for v in p))>1e-9
    # Separabilidad de ensenar: alinear un subconjunto S de nodos reduce A en sum_{v en S} nu D_v
    S=[v for v in p if random.random()<0.5]
    q2={v:(dict(p[v]) if v in S else q[v]) for v in q}
    Q2=dict(walks(succ,q2)); A2=sum(x*L2(x/Q2[w]) for w,x in P.items())
    bad['ensenar separable']+= abs((A-A2)-sum(nu[v]*Dv[v] for v in S))>1e-9
print("fallos en",T,"grafos:",dict(bad))

# ---- Ejemplo de la semana: desalineación de etiquetas (error silencioso) ----
exec(open('ejemplo_semana.py').read().split("out={}")[0])
muR=dict(mu); 
for w in muR:
    if w[2]=='b':   # el receptor lee "manana es X" como "hoy es X"
        muR[w]={'Mar\'':'martes',"Mie'":'miercoles','Jue':'jueves'}[w[3]]
err=sum(P[w] for w in P if mu[w]!=muR[w])
print("error semantico silencioso con etiquetas desalineadas en b: %.3f; costo en bits: 0"%err)
nu={'r':1,'c':1,'a':0.6,'b':0.4}
print("receptor sin estadisticas: %.3f bits = sum nu log2 grado = %.3f"%(sum(P[w]*-L2(math.prod(1/len(p[v]) for v in w[:-1])) for w in P), sum(nu[v]*L2(len(p[v])) for v in nu)))
