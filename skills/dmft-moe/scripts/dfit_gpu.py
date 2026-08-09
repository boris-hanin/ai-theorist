"""1/sqrt(D) demonstration, GPU-native. Reference-free: the large-D limit of the
Mean ODE is a mean-field over the D embedding coordinates, so a coordinate
population average -- the stream kernel (1/D)||h^L||^2 -- FLUCTUATES at 1/sqrt(D).
A fluctuation needs no reference, which is what killed the three earlier attempts.
"""
import torch, numpy as np, math, sys, json, time
dev='cuda'; torch.set_default_dtype(torch.float64)

def run(D,M,L,E,S,X,Y,steps,eta,kappa=0.25):
    a=max(1,int(round(kappa*E))); cL=1.0/(L*M); d0=X.shape[1]
    g=torch.Generator(device=dev).manual_seed(1234)
    rn=lambda *s: torch.randn(*s,generator=g,device=dev)
    U=[rn(S,E,D,M)*D**-0.5 for _ in range(L)]
    W=[rn(S,E,M,D)*1.0     for _ in range(L)]
    R=[rn(S,D,E)*D**-0.5   for _ in range(L)]
    b=[rn(S,E)             for _ in range(L)]
    We=rn(S,d0,D)/math.sqrt(d0); wo=rn(S,D)/D
    ps=U+W+R+[wo]
    lrs=([L*M*a/D*eta]*L)+([L*M*a*D*eta]*L)+([L*a*math.sqrt(M)/D*eta]*L)+[eta/D]
    def fwd():
        h=torch.einsum('pk,skd->spd',X,We)                     # (S,P,D)
        for l in range(L):
            gate=torch.sigmoid(torch.einsum('spd,sde->spe',h,R[l]))
            with torch.no_grad():
                q=gate+b[l].unsqueeze(1)
                idx=q.topk(a,dim=-1).indices
                mask=torch.zeros_like(q).scatter_(-1,idx,1.0)
            z=torch.einsum('spd,sedm->sepm',h,U[l])
            Eo=torch.einsum('sepm,semd->sepd',torch.tanh(z),W[l])
            h=h+cL*((gate*mask).permute(0,2,1).unsqueeze(-1)*Eo).sum(1)/a
        return h
    if steps:
        for p in ps: p.requires_grad_(True)
        for _ in range(steps):
            h=fwd(); f=torch.einsum('spd,sd->sp',h,wo)
            loss=(0.5*(f-Y).pow(2).mean(-1)).sum()
            gs=torch.autograd.grad(loss,ps)
            with torch.no_grad():
                for p,gr,lr in zip(ps,gs,lrs): p-=lr*gr
        for p in ps: p.requires_grad_(False)
    with torch.no_grad():
        h=fwd()
        return h.pow(2).mean(dim=(1,2)).cpu().numpy()          # (S,) kernel per seed

g=torch.Generator(device=dev).manual_seed(4242)
X=torch.randn(16,8,generator=g,device=dev); Y=torch.randn(16,generator=g,device=dev)
S=int(sys.argv[1]); Ds=[16,32,64,128,256,512,1024,2048,4096]
out={}
for tag,aM,steps in [('init  aM=512',512,0),('init  aM=4096',4096,0),
                     ('train aM=512',512,8),('train aM=4096',4096,8)]:
    sp=[];t0=time.time()
    for D in Ds:
        M=max(2,aM//4); chunk=max(1,min(S,int(2.0e8/(16*D*M))))
        v=[]
        for i in range(0,S,chunk):
            v.append(run(D,M,4,16,min(chunk,S-i),X,Y,steps,0.3))
        sp.append(float(np.std(np.concatenate(v),ddof=1)))
    sa=float(np.polyfit(np.log(Ds),np.log(sp),1)[0])
    st=float(np.polyfit(np.log(Ds[-5:]),np.log(sp[-5:]),1)[0])
    out[tag]={'D':Ds,'spread':sp,'slope_all':sa,'slope_tail':st}
    print('%-14s %s'%(tag,' '.join('%.3e'%x for x in sp)),flush=True)
    print('%-14s slope(all) %+.4f  slope(D>=%d) %+.4f  pred -0.5000   [%.0fs]'%('',sa,Ds[-5],st,time.time()-t0),flush=True)
json.dump(out,open('/tmp/dfit_out.json','w'))
print('DONE',flush=True)
