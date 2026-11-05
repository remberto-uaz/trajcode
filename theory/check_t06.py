"""T06: verificación numérica del Teorema 1 (códigos nativos, adaptados, memoria mínima, bits-back)."""
import random, math
from collections import defaultdict
random.seed(5)
def rand_dag(n,maxout,nmean):
    succ={v:[] for v in range(n)}
    for v in range(n-1):
        k=random.randint(1,maxout) if v<n-3 else random.randint(0,maxout)
        c=list(range(v+1,n)); random.shuffle(c); succ[v]=sorted(c[:k])
    p={v:{u:w/sum(ws) for u,w in zip(s,ws)} for v,s in succ.items() if s for ws in [[random.random()+.05 for _ in s]]}
    lab={v:random.randrange(nmean) for v in range(n)}
    return succ,p,lab
def walks(succ,p,v=0,pre=(0,),pr=1.0):
    if not succ[v]: yield pre,pr; return
    for u,q in p[v].items(): yield from walks(succ,p,u,pre+(u,),pr*q)
L2=math.log2; worst={'adapt':0,'classes_ok':True,'bb':0}; maxN=0; nodeMarkov=0; T=400
for t in range(T):
    succ,p,lab=rand_dag(random.randint(6,12),3,random.randint(2,4))
    P=dict(walks(succ,p)); mu={w:lab[w[-1]] for w in P}
    PM=defaultdict(float)
    for w,x in P.items(): PM[mu[w]]+=x
    HM=-sum(x*L2(x) for x in PM.values())
    f={m:max((w for w in P if mu[w]==m),key=P.get) for m in PM}
    Qf={f[m]:PM[m] for m in PM}
    # (c) codigo adaptado: condicionales secuenciales de Q_f dado el prefijo
    def cond_cost(w):
        c=0.0
        for k in range(1,len(w)):
            pre=w[:k]; num=sum(q for x,q in Qf.items() if x[:k+1]==w[:k+1]); den=sum(q for x,q in Qf.items() if x[:k]==pre)
            c+=-L2(num/den)
        return c
    La=sum(PM[m]*cond_cost(f[m]) for m in PM)
    worst['adapt']=max(worst['adapt'],abs(La-HM))
    # Prop. memoria minima: clases de prefijos por nodo = leyes futuras distintas
    fut=defaultdict(set)
    for x in Qf:
        for k in range(1,len(x)):
            pre=x[:k]; den=sum(q for y,q in Qf.items() if y[:k]==pre)
            law=tuple(sorted((y[k:],round(q/den,12)) for y,q in Qf.items() if y[:k]==pre))
            fut[pre[-1]].add(law)
    N=max(len(s) for s in fut.values()); maxN=max(maxN,N); nodeMarkov+= (N==1)
    # (d) bits-back ideal: costo neto esperado = E[-log P(pi)] - E[-log P(pi|m)] = H(M)
    net=sum(x*(-L2(x) + L2(x/PM[mu[w]])) for w,x in P.items())
    worst['bb']=max(worst['bb'],abs(net-HM))
print("(c) |L_adaptado - H(M)| max:",worst['adapt'])
print("(d) |costo neto bits-back - H(M)| max:",worst['bb'])
print("Memoria minima: max_v N_v =",maxN,"; grafos donde Q_f es markoviano por nodo:",nodeMarkov,"de",T)
