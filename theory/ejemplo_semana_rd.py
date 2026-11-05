"""R(D) geodésica del ejemplo de la semana (distancia en el ciclo semanal restringida a lun-mar-mie)."""
import numpy as np
exec(open('check_t08.py').read().split("# ---------- (1)")[0])
p=np.array([0.5,0.3,0.2]); d=np.array([[0,1,2],[1,0,1],[2,1,0]],float)
H=lower_hull(quotient_points(p,d))
def R_at(D):
    lo,hi=1e-3,60
    for _ in range(80):
        b=(lo+hi)/2; R,Db,_=BA(p,d,b)
        if Db>D: lo=b
        else: hi=b
    return BA(p,d,(lo+hi)/2)[:2]
print("puntos de cocientes (D,R):",[(round(a,3),round(b,3)) for a,b in sorted(set((round(x,6),round(y,6)) for x,y in quotient_points(p,d)))])
print("envolvente inferior:",[(round(a,3),round(b,3)) for a,b in H])
for D in [0.1,0.2,0.3,0.5]:
    R,Dr=R_at(D); print(f"D={D}: R(D)={R:.3f}  envolvente de cocientes={interp(H,D):.3f}  brecha={interp(H,D)-R:.3f}")
