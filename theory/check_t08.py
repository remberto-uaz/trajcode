"""T08: R(D) geodésica. (1) cocientes vs Blahut-Arimoto; (2) prueba numérica de refinamiento sucesivo (Equitz-Cover)."""
import numpy as np, itertools, math
from scipy.optimize import linprog
rng=np.random.default_rng(0)
def BA(p,d,beta,iters=200000,tol=1e-13):
    q=np.ones(d.shape[1])/d.shape[1]; A=np.exp(-beta*d)
    for it in range(iters):
        W=q*A; W/=W.sum(1,keepdims=True)      # p(xh|x)
        qn=p@W
        if np.max(np.abs(qn-q))<tol: q=qn; break
        q=qn
    W=q*A; W/=W.sum(1,keepdims=True); J=p[:,None]*W
    with np.errstate(divide='ignore',invalid='ignore'):
        L=np.where(J>1e-300,J*np.log2(np.maximum(J,1e-300)/np.maximum(p[:,None]*q[None,:],1e-300)),0.0)
    return L.sum(),np.sum(J*d),J
def tree_dist(depths_parent):
    pass
def ultrametric(n,levels,rng):
    # hojas con etiquetas de ancestros aleatorias por nivel (particiones anidadas)
    labs=[np.zeros(n,int)]
    for l in range(1,levels):
        prev=labs[-1]; new=np.zeros(n,int); c=0
        for g in np.unique(prev):
            idx=np.where(prev==g)[0]; k=rng.integers(1,len(idx)+1)
            cut=np.sort(rng.choice(np.arange(1,len(idx)),size=k-1,replace=False)) if k>1 else []
            for part in np.split(idx,cut): new[part]=c; c+=1
        labs.append(new)
    labs.append(np.arange(n))
    h=len(labs)-1
    d=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                lca=max(l for l in range(len(labs)) if labs[l][i]==labs[l][j])
                d[i,j]=h-lca
    return d
def path_graph(n):
    return np.abs(np.subtract.outer(np.arange(n),np.arange(n))).astype(float)
def cycle(n):
    a=np.abs(np.subtract.outer(np.arange(n),np.arange(n))); return np.minimum(a,n-a).astype(float)
def refinable(J1,J2,tol=1e-6):
    """¿Existe W(xh1|xh2) estocástica con sum_xh2 J2[x,xh2] W[xh2,xh1] = J1[x,xh1]?"""
    n,m=J2.shape; k=J1.shape[1]
    A_eq=[];b_eq=[]
    for x in range(n):
        for a in range(k):
            row=np.zeros(m*k)
            for b in range(m): row[b*k+a]=J2[x,b]
            A_eq.append(row); b_eq.append(J1[x,a])
    for b in range(m):
        row=np.zeros(m*k); row[b*k:(b+1)*k]=1; A_eq.append(row); b_eq.append(1)
    # minimizar residuo L1 con variables de holgura
    A=np.array(A_eq); nE=len(b_eq)
    c=np.r_[np.zeros(m*k),np.ones(2*nE)]
    Aeq=np.c_[A,np.eye(nE),-np.eye(nE)]
    res=linprog(c,A_eq=Aeq,b_eq=b_eq,bounds=[(0,None)]*(m*k+2*nE),method='highs')
    return res.fun
def quotient_points(p,d):
    n=len(p); pts=[]
    def partitions(s):
        if not s: yield []; return
        f=s[0]
        for rest in partitions(s[1:]):
            for i in range(len(rest)):
                yield rest[:i]+[[f]+rest[i]]+rest[i+1:]
            yield [[f]]+rest
    for P in partitions(list(range(n))):
        R=-sum(p[B].sum()*math.log2(p[B].sum()) for B in P)
        D=sum(min(sum(p[i]*d[i,b] for i in B) for b in B) for B in P)
        pts.append((D,R))
    return pts
def lower_hull(pts):
    pts=sorted(set((round(a,12),round(b,12)) for a,b in pts)); H=[]
    for P in pts:
        while len(H)>=2 and (H[-1][0]-H[-2][0])*(P[1]-H[-2][1])-(H[-1][1]-H[-2][1])*(P[0]-H[-2][0])<=0: H.pop()
        H.append(P)
    return H
def interp(H,D):
    for (a,b),(c,e) in zip(H,H[1:]):
        if a<=D<=c: return b+(e-b)*(D-a)/(c-a)
    return H[-1][1]
# ---------- (1) brecha de los cocientes ----------
print("== (1) cociente (envolvente convexa) vs R(D) de Blahut-Arimoto ==")
for name,dist in [("ciclo-6",cycle(6)),("camino-6",path_graph(6)),("ultrametrica-6",ultrametric(6,3,rng))]:
    p=rng.dirichlet(np.ones(6)); H=lower_hull(quotient_points(p,dist))
    gaps=[]
    for beta in [0.5,1,2,3,5]:
        R,D,_=BA(p,dist,beta); gaps.append((round(D,3),round(interp(H,D)-R,3)))
    print(name,"(D, brecha) =",gaps)
# ---------- (2) refinamiento sucesivo ----------
print("== (2) prueba de Equitz-Cover (residuo LP; 0 = refinable) ==")
for name,gen in [("hamming(control)",lambda: 1-np.eye(5)),("ultrametrica",lambda: ultrametric(5,3,rng)),("camino",lambda: path_graph(5)),("ciclo",lambda: cycle(5))]:
    worst=0; fails=0; trials=60
    for t in range(trials):
        dist=gen(); p=rng.dirichlet(np.ones(5))
        b1,b2=sorted(rng.uniform(0.4,5,2))
        _,_,J1=BA(p,dist,b1); _,_,J2=BA(p,dist,b2)
        r=refinable(J1,J2); worst=max(worst,r); fails+= r>1e-5
    print(f"{name:12s} pares no refinables: {fails}/{trials}  residuo max: {worst:.2e}")
