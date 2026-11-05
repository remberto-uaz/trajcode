"""T09c: en un código de camino que sigue la jerarquía, el prefijo hasta la profundidad l revela la clase de Q_l.
Comprueba: (1) costo ideal acumulado hasta l = H(Q_l) exactamente; (2) con un código de Shannon por decisión
(longitudes enteras) el costo esperado hasta l es <= H(Q_l) + l; (3) contraste: un código de Huffman sobre los
significados no revela la clase de nivel 1 en un prefijo fijo."""
import random, math, heapq, itertools
L2=math.log2; random.seed(4)
def rand_tree(h,maxb):
    # devuelve hojas como tuplas de ramas (camino desde la raiz)
    leaves=[()]
    for _ in range(h):
        leaves=[l+(b,) for l in leaves for b in range(random.randint(1,maxb))]
    return leaves
worst1=0; viol2=0; T=500; huff_bad=0; multi=0
for t in range(T):
    h=random.randint(2,4); leaves=rand_tree(h,3)
    w=[random.random()**2+1e-3 for _ in leaves]; z=sum(w); P={l:x/z for l,x in zip(leaves,w)}
    for lev in range(1,h+1):
        cls={}
        for l,x in P.items(): cls[l[:lev]]=cls.get(l[:lev],0)+x
        HQ=-sum(x*L2(x) for x in cls.values())
        # (1) costo ideal: suma de -log de condicionales por decision hasta lev = -log P(clase)
        ideal=0; shannon=0
        for l,x in P.items():
            c=0; cs=0
            for k in range(lev):
                num=sum(y for m,y in P.items() if m[:k+1]==l[:k+1]); den=sum(y for m,y in P.items() if m[:k]==l[:k])
                c+=-L2(num/den); cs+=math.ceil(-L2(num/den)-1e-12)
            ideal+=x*c; shannon+=x*cs
        worst1=max(worst1,abs(ideal-HQ)); viol2+= shannon>HQ+lev+1e-9
    # (3) Huffman sobre significados: ¿los codigos de una misma clase de nivel 1 comparten un prefijo propio?
    heap=[(x,i,[l]) for i,(l,x) in enumerate(P.items())]; heapq.heapify(heap); code={l:'' for l in P}; cnt=itertools.count(len(heap))
    while len(heap)>1:
        a=heapq.heappop(heap); b=heapq.heappop(heap)
        for l in a[2]: code[l]='0'+code[l]
        for l in b[2]: code[l]='1'+code[l]
        heapq.heappush(heap,(a[0]+b[0],next(cnt),a[2]+b[2]))
    groups={}
    for l in P: groups.setdefault(l[:1],[]).append(code[l])
    def lcp(xs): return len(__import__('os').path.commonprefix(xs))
    # la clase es identificable desde un prefijo comun solo si ningun codigo de otra clase empieza igual
    ok=all(not any(code[m].startswith(__import__('os').path.commonprefix(g)) for m in P if m[:1]!=k) for k,g in groups.items() if len(groups)>1)
    huff_bad+= (len(groups)>1 and not ok); multi+= len(groups)>1
print("(1) max |costo ideal hasta l - H(Q_l)|:",worst1)
print("(2) violaciones de costo Shannon <= H(Q_l)+l:",viol2)
print("(3) arboles donde Huffman sobre significados NO revela la clase de nivel 1 con un prefijo:",huff_bad,"de",T)
print("(4) arboles con al menos dos clases de nivel 1 (los unicos en que Huffman puede fallar):",multi,"de",T)
