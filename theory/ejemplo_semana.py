"""Ejemplo pedagógico de theory/00-modelo.tex: todos los números del texto salen de aquí."""
from math import log2
# Red: r -> c (p=1); c -> a 'hoy es' (0.6), c -> b 'manana es' (0.4)
# a -> Lun 0.5, Mar 0.3, Mie 0.2 ; b -> Mar' 0.5, Mie' 0.3, Jue 0.2
p = {'r':{'c':1.0}, 'c':{'a':0.6,'b':0.4},
     'a':{'Lun':0.5,'Mar':0.3,'Mie':0.2}, 'b':{"Mar'":0.5,"Mie'":0.3,'Jue':0.2}}
# significado = dia de HOY
mu = {('r','c','a','Lun'):'lunes', ('r','c','a','Mar'):'martes', ('r','c','a','Mie'):'miercoles',
      ('r','c','b',"Mar'"):'lunes', ('r','c','b',"Mie'"):'martes', ('r','c','b','Jue'):'miercoles'}
def prob(path,q):
    pr=1.0
    for u,v in zip(path,path[1:]): pr*=q[u].get(v,0.0)
    return pr
H=lambda d: -sum(x*log2(x) for x in d if x>0)
P={pi:prob(pi,p) for pi in mu}
PM={}
for pi,x in P.items(): PM[mu[pi]]=PM.get(mu[pi],0)+x
HPi=H(P.values()); HM=H(PM.values())
L0=sum(PM[m]*-log2(max(P[pi] for pi in P if mu[pi]==m)) for m in PM)
gaps={m:log2(PM[m]/max(P[pi] for pi in P if mu[pi]==m)) for m in PM}
out={}
out['P']={'-'.join(k):round(v,4) for k,v in P.items()}
out['PM']={k:round(v,4) for k,v in PM.items()}
out['H(Pi)']=HPi; out['H(M)']=HM; out['H(Pi|M)']=HPi-HM; out['L0']=L0; out['gap']=L0-HM; out['g(m)']=gaps
# costo por decision del camino r-c-a-Lun
out['bits r-c-a-Lun']=[-log2(p['r']['c']),-log2(p['c']['a']),-log2(p['a']['Lun'])]
# Desalineacion 1: receptor con ruteo uniforme en a
q={k:dict(v) for k,v in p.items()}; q['a']={'Lun':1/3,'Mar':1/3,'Mie':1/3}
Q={pi:prob(pi,q) for pi in mu}
A=sum(P[pi]*log2(P[pi]/Q[pi]) for pi in P)
QM={}
for pi,x in Q.items(): QM[mu[pi]]=QM.get(mu[pi],0)+x
Ds=sum(PM[m]*log2(PM[m]/QM[m]) for m in PM)
nu_a=p['c']['a']
out['A1']=A; out['A1 nodo a']=nu_a*sum(p['a'][u]*log2(p['a'][u]/q['a'][u]) for u in p['a'])
out['A1 cota 2d']=nu_a*log2(3); out['A1 D_s']=Ds; out['A1 cond']=A-Ds
# Desalineacion 2: receptor sin la arista b->Jue
out['A2']='infinito: el camino r-c-b-Jue tiene probabilidad 0 para el receptor'
# Distorsion geodesica: dias en ciclo semanal, d(lun,mar)=d(mar,mie)=1, d(lun,mie)=2
pm=[PM['lunes'],PM['martes'],PM['miercoles']]
out['R cociente {lun,mar}|{mie}, D<=1']=H([pm[0]+pm[1],pm[2]])
out['R cociente {lun}|{mar,mie}, D<=1']=H([pm[0],pm[1]+pm[2]])
out['Edist cociente {lun,mar}->mar, mie->mie']=pm[0]*1
for k,v in out.items(): print(k,':',v if not isinstance(v,float) else round(v,4))
