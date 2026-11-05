"""T06b: memoria mínima cuando el significado depende de la historia (contexto) y no solo del nodo final."""
import random, math
from collections import defaultdict
exec(open('check_t06.py').read().split("L2=math.log2")[0])
L2=math.log2
def analyze(mu_rule,T=400,seed=9):
    random.seed(seed); maxN=0; needmem=0; excess=[]
    for t in range(T):
        succ,p,lab=rand_dag(random.randint(6,12),3,random.randint(2,4))
        P=dict(walks(succ,p)); mu={w:mu_rule(w,lab) for w in P}
        PM=defaultdict(float)
        for w,x in P.items(): PM[mu[w]]+=x
        f={m:max((w for w in P if mu[w]==m),key=P.get) for m in PM}
        Qf={f[m]:PM[m] for m in PM}
        laws=defaultdict(set); prefs=defaultdict(set)
        for x in Qf:
            for k in range(1,len(x)):
                pre=x[:k]; den=sum(q for y,q in Qf.items() if y[:k]==pre)
                laws[pre[-1]].add(tuple(sorted((y[k:],round(q/den,12)) for y,q in Qf.items() if y[:k]==pre)))
        N=max(len(s) for s in laws.values()); maxN=max(maxN,N); needmem+=(N>1)
        # costo de un codigo adaptado SIN memoria (kernel por nodo = promedio de Q_f en el nodo)
        cnt=defaultdict(lambda: defaultdict(float))
        for x,q in Qf.items():
            for a,b in zip(x,x[1:]): cnt[a][b]+=q
        ker={a:{b:c/sum(d.values()) for b,c in d.items()} for a,d in cnt.items()}
        L=sum(q*sum(-L2(ker[a][b]) for a,b in zip(x,x[1:])) for x,q in Qf.items())
        HM=-sum(x*L2(x) for x in PM.values()); excess.append(L-HM)
    return maxN,needmem,max(excess),min(excess)
print("significado = etiqueta del nodo final        :",analyze(lambda w,lab: lab[w[-1]]))
print("significado = (etiqueta 2do nodo + final) mod:",analyze(lambda w,lab: (lab[w[1]]+lab[w[-1]])%3))
