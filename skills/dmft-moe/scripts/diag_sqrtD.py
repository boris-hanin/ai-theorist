"""Why does 1/sqrt(D) fail after training? DMFT says: the kernel fluctuation has
a second, SHARED source -- the error trajectory Delta(t), a population average
fluctuating at 1/sqrt(D) that enters every coordinate-site coherently. Its
contribution is (dK/dDelta)*dDelta, so the law holds at FIXED susceptibility
(the F5 Delta-loop gain) and fails if models at different D sit at different
points along training after a fixed step count.

Three predictions, in increasing sharpness:
  P1  training progress (loss at fixed step count) is D-dependent
  P2  the deviation from -1/2 GROWS with the training horizon
  P3  matching by LOSS instead of by step count RESTORES -1/2
"""
import torch, numpy as np, math, json, time
dev='cuda'; torch.set_default_dtype(torch.float64)
def log(*a): print('[%s]'%time.strftime('%H:%M:%S'),*a,flush=True)
g0=torch.Generator(device=dev).manual_seed(4242)
X=torch.randn(16,8,generator=g0,device=dev); Y=torch.randn(16,generator=g0,device=dev)

def run(D,S,steps,eta=0.3,aM=512,L=4,E=16,a=4,target=None):
    """Return (kernel per seed, loss per seed, steps actually taken per seed)."""
    M=max(2,aM//4); cL=1./(L*M); kap=a/E
    per=max(1,int(8e7/max(1,L*E*D*M))); K=[];LO=[];NS=[]; done=0
    while done<S:
        s=min(per,S-done)
        g=torch.Generator(device=dev).manual_seed(555+done)
        rn=lambda *z: torch.randn(*z,generator=g,device=dev)
        U=[rn(s,E,D,M)*D**-0.5 for _ in range(L)]; W=[rn(s,E,M,D) for _ in range(L)]
        R=[rn(s,D,E)*D**-0.5 for _ in range(L)]; b=[rn(s,E) for _ in range(L)]
        We=rn(s,8,D)/math.sqrt(8); wo=rn(s,D)/D
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
                h=h+cL*((gt*m).permute(0,2,1).unsqueeze(-1)*torch.einsum('sepm,semd->sepd',torch.tanh(z),W[l])).sum(1)/a
            return h
        for p in ps: p.requires_grad_(True)
        n=0
        for it in range(steps):
            f=torch.einsum('spd,sd->sp',fwd(),wo)
            lo=(0.5*(f-Y).pow(2).mean(-1))
            if target is not None and float(lo.mean())<=target: break
            gs=torch.autograd.grad(lo.sum(),ps)
            with torch.no_grad():
                for p,gr,lr in zip(ps,gs,lrs): p-=lr*gr
            n=it+1
        with torch.no_grad():
            h=fwd(); f=torch.einsum('spd,sd->sp',h,wo)
            K.append(h.pow(2).mean(dim=(1,2)).cpu().numpy())
            LO.append((0.5*(f-Y).pow(2).mean(-1)).cpu().numpy())
        for p in ps: p.requires_grad_(False)
        del U,W,R,b,We,wo,ps; torch.cuda.empty_cache(); done+=s; NS.append(n)
    return np.concatenate(K), np.concatenate(LO), float(np.mean(NS))

def sl(x,y): return float(np.polyfit(np.log(x),np.log(y),1)[0])
Ds=[32,64,128,256,512,1024,2048,4096]; S=512; OUT={}

log('P1: is training progress D-dependent at a fixed step count?')
for st in (0,8,24):
    los=[];sp=[]
    for D in Ds:
        k,l,_=run(D,S,st); los.append(float(l.mean())); sp.append(float(k.std(ddof=1)))
    OUT['P1_steps%d'%st]={'D':Ds,'loss':los,'spread':sp,'slope':sl(Ds,sp)}
    log(' steps=%-3d loss %s'%(st,' '.join('%.4f'%v for v in los)))
    log('           kernel-spread slope %+.4f'%sl(Ds,sp))
    json.dump(OUT,open('/home/ubuntu/diag_out.json','w'),indent=1)

log('P2: does the deviation from -1/2 GROW with the horizon?')
for st in (2,4,8,16,32,64):
    sp=[run(D,S,st)[0].std(ddof=1) for D in Ds]
    OUT['P2_h%d'%st]={'steps':st,'slope':sl(Ds,sp)}
    log(' horizon %-3d  slope %+.4f  (deviation %+.4f)'%(st,sl(Ds,sp),sl(Ds,sp)+0.5))
    json.dump(OUT,open('/home/ubuntu/diag_out.json','w'),indent=1)

log('P3: match by LOSS instead of by step count -- does -1/2 come back?')
for tgt in (0.40,0.30,0.20):
    sp=[];ns=[]
    for D in Ds:
        k,l,n=run(D,S,200,target=tgt); sp.append(float(k.std(ddof=1))); ns.append(n)
    OUT['P3_loss%.2f'%tgt]={'D':Ds,'spread':sp,'slope':sl(Ds,sp),'steps_used':ns}
    log(' target loss %.2f  steps used %s'%(tgt,' '.join('%.0f'%v for v in ns)))
    log('                   slope %+.4f   (fixed-step gave -0.45)'%sl(Ds,sp))
    json.dump(OUT,open('/home/ubuntu/diag_out.json','w'),indent=1)
log('DIAG DONE')
