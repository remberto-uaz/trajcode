"""Busca el contraejemplo mínimo: significado dependiente del contexto que obliga a llevar memoria."""
import itertools, math
L2=math.log2
pa=0.6; px={'t1':0.5,'t2':0.3,'t3':0.2}
walks={(br,t):(pa if br=='a' else 1-pa)*q for br in 'ab' for t,q in px.items()}
best=None
for labels in itertools.product('ABC',repeat=6):
    mu=dict(zip(walks,labels))
    if len(set(labels))<3: continue
    PM={}
    for w,x in walks.items(): PM[mu[w]]=PM.get(mu[w],0)+x
    f={m:max((w for w in walks if mu[w]==m),key=walks.get) for m in PM}
    Qf={f[m]:PM[m] for m in PM}
    laws={}
    for br in 'ab':
        tot=sum(q for w,q in Qf.items() if w[0]==br)
        if tot>0: laws[br]=tuple(sorted((w[1],round(q/tot,9)) for w,q in Qf.items() if w[0]==br))
    if len(laws)==2 and laws['a']!=laws['b']:
        # costo del codigo adaptado sin memoria: kernel en x = mezcla
        mix={}
        for w,q in Qf.items(): mix[w[1]]=mix.get(w[1],0)+q
        pb={br:sum(q for w,q in Qf.items() if w[0]==br) for br in 'ab'}
        L=sum(q*(-L2(pb[w[0]]) - L2(mix[w[1]])) for w,q in Qf.items())
        HM=-sum(x*L2(x) for x in PM.values())
        cand=(L-HM,labels,PM,f,laws)
        if best is None or cand[0]>best[0]: best=cand
ex,labels,PM,f,laws=best
print("etiquetas (a,t1)(a,t2)(a,t3)(b,t1)(b,t2)(b,t3):",labels)
print("P_M:",{k:round(v,3) for k,v in PM.items()}); print("f*:",f); print("leyes futuras en x:",laws)
print("exceso del codigo adaptado sin memoria (bits):",round(ex,4))

# --- Números exactos del Ejemplo 2 (polisemia) para theory/01-memoria.tex ---
labels=('A','A','B','A','C','C'); mu=dict(zip(walks,labels))
PM={}
for w,x in walks.items(): PM[mu[w]]=PM.get(mu[w],0)+x
HM=-sum(x*L2(x) for x in PM.values())
HPi=-sum(x*L2(x) for x in walks.values())
L0=sum(PM[m]*-L2(max(walks[w] for w in walks if mu[w]==m)) for m in PM)
sel={m:[w for w in walks if mu[w]==m] for m in PM}
bestL=None
for choice in itertools.product(*[sel[m] for m in sorted(PM)]):
    f=dict(zip(sorted(PM),choice)); Qf={f[m]:PM[m] for m in PM}
    pb={br:sum(q for w,q in Qf.items() if w[0]==br) for br in 'ab'}
    mix={}
    for w,q in Qf.items(): mix[w[1]]=mix.get(w[1],0)+q
    L=sum(q*(-L2(pb[w[0]]) - L2(mix[w[1]])) for w,q in Qf.items() if q>0)
    nlaws=len({tuple(sorted((w[1],round(q/pb[br],9)) for w,q in Qf.items() if w[0]==br)) for br in 'ab' if pb[br]>0})
    if bestL is None or L<bestL[0]: bestL=(L,f,nlaws)
print("H(M)=%.4f  H(Pi)=%.4f  H(Pi|M)=%.4f"%(HM,HPi,HPi-HM))
print("nativo L0=%.4f (exceso %.4f)"%(L0,L0-HM))
print("adaptado sin memoria, mejor seleccion: L=%.4f (exceso %.4f), f=%s, leyes distintas en x=%d"%(bestL[0],bestL[0]-HM,bestL[1],bestL[2]))
print("adaptado con 2 estados: L=H(M)=%.4f"%HM)
