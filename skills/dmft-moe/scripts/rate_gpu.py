"""C^{-1/6} rate test for the MoE Mean-ODE limit (derivations/09 §4).

Reference-free by construction: along the optimal-shape path every error term is
Theta(C^{-1/6}), so the distance between the C and 4C ensembles is too --
    E_diff(C)^2 = |mu_C - mu_4C|^2 + var_C + var_4C   ~  C^{-1/3}
which needs no limit model. (A limit reference is impossible here: C^{-1/6} is so
slow that a reference 10x better than C=1e7 would need C_ref ~ 1e13.)

Shape at budget C (derivations/09 §4, as corrected):  D ~ C^(1/3), L ~ C^(1/6),
aM = C/(L D).  All three error terms are then the same order.
"""
import torch, numpy as np, math, sys, json, time
dev='cuda'; torch.set_default_dtype(torch.float64)

def shape(C, a=4):
    L = max(2, int(round(C ** (1/6.0))))
    D = max(4, int(round(C ** (1/3.0))))
    M = max(2, int(round(C / (L * D * a))))
    return L, D, M

def ens(C, S, X, Y, steps=8, eta=0.3, E=16, a=4, sd0=0):
    """Return (S,P) outputs for S independent draws at budget C."""
    L, D, M = shape(C, a)
    cL = 1.0/(L*M); d0 = X.shape[1]
    outs = []
    # chunk seeds so the batched params fit in memory
    per = max(1, int(2.5e8 / max(1, E*D*M)))
    done = 0
    while done < S:
        s = min(per, S-done)
        g = torch.Generator(device=dev).manual_seed(9000+sd0+done)
        rn = lambda *z: torch.randn(*z, generator=g, device=dev)
        U=[rn(s,E,D,M)*D**-0.5 for _ in range(L)]
        W=[rn(s,E,M,D)         for _ in range(L)]
        R=[rn(s,D,E)*D**-0.5   for _ in range(L)]
        b=[rn(s,E)             for _ in range(L)]
        We=rn(s,d0,D)/math.sqrt(d0); wo=rn(s,D)/D
        ps=U+W+R+[wo]
        lrs=([L*M*a/D*eta]*L)+([L*M*a*D*eta]*L)+([L*a*math.sqrt(M)/D*eta]*L)+[eta/D]
        def fwd():
            h=torch.einsum('pk,skd->spd',X,We)
            for l in range(L):
                gt=torch.sigmoid(torch.einsum('spd,sde->spe',h,R[l]))
                with torch.no_grad():
                    q=gt+b[l].unsqueeze(1)
                    m=torch.zeros_like(q).scatter_(-1,q.topk(a,-1).indices,1.0)
                z=torch.einsum('spd,sedm->sepm',h,U[l])
                Eo=torch.einsum('sepm,semd->sepd',torch.tanh(z),W[l])
                h=h+cL*((gt*m).permute(0,2,1).unsqueeze(-1)*Eo).sum(1)/a
            return h
        for p in ps: p.requires_grad_(True)
        for _ in range(steps):
            f=torch.einsum('spd,sd->sp',fwd(),wo)
            gs=torch.autograd.grad((0.5*(f-Y).pow(2).mean(-1)).sum(),ps)
            with torch.no_grad():
                for p,gr,lr in zip(ps,gs,lrs): p-=lr*gr
        with torch.no_grad():
            outs.append(torch.einsum('spd,sd->sp',fwd(),wo).cpu())
        for p in ps: p.requires_grad_(False)
        del U,W,R,b,We,wo,ps; torch.cuda.empty_cache()
        done += s
    return torch.cat(outs).numpy()

g=torch.Generator(device=dev).manual_seed(4242)
X=torch.randn(16,8,generator=g,device=dev); Y=torch.randn(16,generator=g,device=dev)
Cs=[1e3,4e3,1.6e4,6.4e4,2.56e5,1.024e6,4.096e6,1.6384e7]
S=int(sys.argv[1]) if len(sys.argv)>1 else 24
F={}
for C in Cs:
    t0=time.time(); L,D,M=shape(C)
    F[C]=ens(C,S,X,Y,sd0=int(math.log10(C)*1000))
    print('C=%.3e  L=%-3d D=%-5d M=%-6d aM=%-7d  actualC=%.2e  [%.0fs]'
          %(C,L,D,M,4*M,L*4*M*D,time.time()-t0),flush=True)
res=[]
for i in range(len(Cs)-1):
    A,B=F[Cs[i]],F[Cs[i+1]]
    d=math.sqrt(float(((A.mean(0)-B.mean(0))**2).mean()+A.var(0,ddof=1).mean()+B.var(0,ddof=1).mean()))
    res.append((Cs[i],d)); print('  C=%.3e -> %.3e   E_diff %.5e'%(Cs[i],Cs[i+1],d),flush=True)
x=np.log([r[0] for r in res]); y=np.log([r[1] for r in res])
sl=float(np.polyfit(x,y,1)[0]); slt=float(np.polyfit(x[-4:],y[-4:],1)[0])
print('\nslope(all)  %+.4f   slope(last4) %+.4f   PREDICTED -0.1667'%(sl,slt),flush=True)
json.dump({'C':[r[0] for r in res],'Ediff':[r[1] for r in res],'slope':sl,'slope_tail':slt},
          open('/tmp/rate_out.json','w'))
print('DONE',flush=True)
