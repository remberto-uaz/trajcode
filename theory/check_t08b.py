"""T08b: ¿en qué subclase ultramétrica se cumple el refinamiento sucesivo?"""
import numpy as np
exec(open('check_t08.py').read().split("# ---------- (1)")[0])
rng=np.random.default_rng(1)
def balanced(br,levels):
    n=br**levels; d=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                k=0
                while (i//br**(levels-1-k))==(j//br**(levels-1-k)): k+=1
                d[i,j]=levels-k
    return d
def test(name,gen_d,gen_p,trials=60):
    fails=0; worst=0; supp_fail=[]
    for t in range(trials):
        dist=gen_d(); p=gen_p(dist.shape[0])
        b1,b2=sorted(rng.uniform(0.4,5,2))
        _,D1,J1=BA(p,dist,b1); _,D2,J2=BA(p,dist,b2)
        r=refinable(J1,J2); worst=max(worst,r)
        if r>1e-5:
            fails+=1; supp_fail.append((int((J1.sum(0)>1e-9).sum()),int((J2.sum(0)>1e-9).sum())))
    print(f"{name:42s} no refinables {fails:2d}/{trials}  residuo max {worst:.1e}  soportes(q1,q2) en fallos: {supp_fail[:6]}")
n=8
test("ultrametrica balanceada 2x3, p uniforme",lambda: balanced(2,3),lambda n: np.ones(n)/n)
test("ultrametrica balanceada 2x3, p aleatoria",lambda: balanced(2,3),lambda n: rng.dirichlet(np.ones(n)))
test("ultrametrica aleatoria (6 hojas), p uniforme",lambda: ultrametric(6,3,rng),lambda n: np.ones(n)/n)
test("ultrametrica aleatoria (6 hojas), p aleatoria",lambda: ultrametric(6,3,rng),lambda n: rng.dirichlet(np.ones(n)))
test("hamming (control), p aleatoria",lambda: 1-np.eye(6),lambda n: rng.dirichlet(np.ones(n)))
