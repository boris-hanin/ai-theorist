"""Overnight suite for the MoE Mean-ODE limit. Self-contained (no local imports).

Robustness: every experiment is wrapped, results are written to JSON after EVERY
sub-result, and the driver runs experiments in descending order of value so that
a spot preemption still leaves the important ones done.

  E1  C^{-1/6} rate, wide ladder, high seeds, BOTH eta_bias arms, and the
      load-movement diagnostic folded in (answers the open caveat).
  E2  HP transfer across all five dials WITH load balancing on.
  E3  1/sqrt(D) pushed to D = 16384.
  E4  lazy branch alpha* = (ML)^{1/4} against a large linearised reference.
"""
import torch, numpy as np, math, json, time, os, sys, traceback
dev='cuda'; torch.set_default_dtype(torch.float64)
OUT='/home/ubuntu/big_out.json'; RES={}
def save():
    tmp=OUT+'.tmp'; json.dump(RES,open(tmp,'w'),indent=1); os.replace(tmp,OUT)
def log(*a):
    print('[%s]'%time.strftime('%H:%M:%S'),*a,flush=True)

def shape(C,a=4):
    L=max(2,int(round(C**(1/6.)))); D=max(4,int(round(C**(1/3.))))
    M=max(2,int(round(C/(L*D*a)))); return L,D,M

def make(S,L,D,M,E,d0,g):
    rn=lambda *z: torch.randn(*z,generator=g,device=dev)
    return dict(U=[rn(S,E,D,M)*D**-0.5 for _ in range(L)],
                W=[rn(S,E,M,D) for _ in range(L)],
                R=[rn(S,D,E)*D**-0.5 for _ in range(L)],
                b=[rn(S,E) for _ in range(L)],
                We=rn(S,d0,D)/math.sqrt(d0), wo=rn(S,D)/D)

def run_ens(C,S,X,Y,steps=8,eta=0.3,E=16,a=4,sd0=0,eta_bias=1.0,mem=8e7):
    L,D,M=shape(C,a); cL=1./(L*M); kap=a/E; d0=X.shape[1]
    outs=[]; imb0=[]; imb1=[]; done=0
    per=max(1,int(mem/max(1,L*E*D*M)))
    while done<S:
        s=min(per,S-done)
        g=torch.Generator(device=dev).manual_seed(9000+sd0+done)
        P=make(s,L,D,M,E,d0,g)
        U,W,R,b,We,wo=P['U'],P['W'],P['R'],P['b'],P['We'],P['wo']
        ps=U+W+R+[wo]
        lrs=([L*M*a/D*eta]*L)+([L*M*a*D*eta]*L)+([L*a*math.sqrt(M)/D*eta]*L)+[eta/D]
        loads=[None]*L
        def fwd():
            h=torch.einsum('pk,skd->spd',X,We)
            for l in range(L):
                gt=torch.sigmoid(torch.einsum('spd,sde->spe',h,R[l]))
                with torch.no_grad():
                    q=gt+b[l].unsqueeze(1)
                    m=torch.zeros_like(q).scatter_(-1,q.topk(a,-1).indices,1.0)
                    loads[l]=m.mean(1)
                z=torch.einsum('spd,sedm->sepm',h,U[l])
                Eo=torch.einsum('sepm,semd->sepd',torch.tanh(z),W[l])
                h=h+cL*((gt*m).permute(0,2,1).unsqueeze(-1)*Eo).sum(1)/a
            return h
        with torch.no_grad():
            fwd(); imb0.append(float(max((l_-kap).abs().max() for l_ in loads)))
        for p in ps: p.requires_grad_(True)
        for _ in range(steps):
            f=torch.einsum('spd,sd->sp',fwd(),wo)
            gs=torch.autograd.grad((0.5*(f-Y).pow(2).mean(-1)).sum(),ps)
            with torch.no_grad():
                for p,gr,lr in zip(ps,gs,lrs): p-=lr*gr
                for l in range(L):
                    if loads[l] is not None: b[l]-=eta_bias*(loads[l]-kap)
        with torch.no_grad():
            h=fwd(); outs.append(torch.einsum('spd,sd->sp',h,wo).cpu())
            imb1.append(float(max((l_-kap).abs().max() for l_ in loads)))
        for p in ps: p.requires_grad_(False)
        del U,W,R,b,We,wo,ps,P,loads; torch.cuda.empty_cache()
        done+=s
    return torch.cat(outs).numpy(), float(np.mean(imb0)), float(np.mean(imb1))

def slope(x,y): return float(np.polyfit(np.log(x),np.log(y),1)[0])

g0=torch.Generator(device=dev).manual_seed(4242)
X=torch.randn(16,8,generator=g0,device=dev); Y=torch.randn(16,generator=g0,device=dev)

# ---------------- E1 : the rate, both arms, with load diagnostics ------------
Cs=[1e3,4e3,1.6e4,6.4e4,2.56e5,1.024e6,4.096e6,1.6384e7,6.5536e7,2.621e8]
Ss=[512,512,512,384,256,192,128,96,48,24]
for eb in (1.0,0.0):
    key='E1_rate_eta_bias_%g'%eb; RES[key]={'C':[],'Ediff':[],'imb_before':[],'imb_after':[]}
    F={}
    for C,S in zip(Cs,Ss):
        try:
            t0=time.time(); f,i0,i1=run_ens(C,S,X,Y,sd0=int(math.log10(C)*1000),eta_bias=eb)
            F[C]=f; L,D,M=shape(C)
            RES[key]['imb_before'].append(i0); RES[key]['imb_after'].append(i1)
            log('E1 eb=%g C=%.3e L=%d D=%d M=%d S=%d imb %.3f->%.3f [%.0fs]'%(eb,C,L,D,M,S,i0,i1,time.time()-t0))
            save()
        except Exception as e:
            log('E1 FAIL C=%.3e : %r'%(C,e)); torch.cuda.empty_cache(); break
    ks=sorted(F)
    for i in range(len(ks)-1):
        A,B=F[ks[i]],F[ks[i+1]]
        d=math.sqrt(float(((A.mean(0)-B.mean(0))**2).mean()+A.var(0,ddof=1).mean()+B.var(0,ddof=1).mean()))
        RES[key]['C'].append(ks[i]); RES[key]['Ediff'].append(d)
    if len(RES[key]['C'])>2:
        RES[key]['slope']=slope(RES[key]['C'],RES[key]['Ediff'])
        RES[key]['slope_tail']=slope(RES[key]['C'][-4:],RES[key]['Ediff'][-4:])
        log('E1 eb=%g  slope %+.4f  tail %+.4f  (pred -0.1667)'%(eb,RES[key]['slope'],RES[key]['slope_tail']))
    save(); del F; torch.cuda.empty_cache()

# ---------------- E3 : 1/sqrt(D) to D = 16384 --------------------------------
try:
    RES['E3_sqrtD']={}
    for tag,aM,steps in [('init',512,0),('trained',512,8)]:
        Ds=[32,64,128,256,512,1024,2048,4096,8192,16384]; sp=[]
        for D in Ds:
            M=max(2,aM//4); L=4; E=16; a=4; S=192
            per=max(1,int(8e7/max(1,L*E*D*M))); vals=[]; done=0
            while done<S:
                s=min(per,S-done)
                g=torch.Generator(device=dev).manual_seed(555+done)
                P=make(s,L,D,M,E,8,g)
                with torch.no_grad():
                    h=torch.einsum('pk,skd->spd',X,P['We'])
                    for l in range(L):
                        gt=torch.sigmoid(torch.einsum('spd,sde->spe',h,P['R'][l]))
                        q=gt+P['b'][l].unsqueeze(1)
                        m=torch.zeros_like(q).scatter_(-1,q.topk(a,-1).indices,1.0)
                        z=torch.einsum('spd,sedm->sepm',h,P['U'][l])
                        Eo=torch.einsum('sepm,semd->sepd',torch.tanh(z),P['W'][l])
                        h=h+(1./(L*M))*((gt*m).permute(0,2,1).unsqueeze(-1)*Eo).sum(1)/a
                    vals.append(h.pow(2).mean(dim=(1,2)).cpu().numpy())
                del P; torch.cuda.empty_cache(); done+=s
            sp.append(float(np.std(np.concatenate(vals),ddof=1)))
            log('E3 %s D=%-6d spread %.4e'%(tag,D,sp[-1])); save()
        RES['E3_sqrtD'][tag]={'D':Ds,'spread':sp,'slope':slope(Ds,sp),'slope_tail':slope(Ds[-5:],sp[-5:])}
        log('E3 %s slope %+.4f tail %+.4f (pred -0.5)'%(tag,RES['E3_sqrtD'][tag]['slope'],RES['E3_sqrtD'][tag]['slope_tail']))
        save()
except Exception as e:
    log('E3 FAIL %r'%e); RES['E3_error']=repr(e); save()

# ---------------- E2 : HP transfer, all five dials, balancing ON -------------
def opt_lr(L,D,M,E,kap,S=48,steps=8,grid=None,eb=1.0):
    grid=grid if grid is not None else np.logspace(-1.0,2.4,18)
    losses=[]
    for eta in grid:
        a=max(1,int(round(kap*E))); cL=1./(L*M); tot=0.;n=0; done=0
        per=max(1,int(8e7/max(1,L*E*D*M)))
        while done<S:
            s=min(per,S-done)
            g=torch.Generator(device=dev).manual_seed(31+done)
            P=make(s,L,D,M,E,8,g); U,W,R,b,We,wo=P['U'],P['W'],P['R'],P['b'],P['We'],P['wo']
            ps=U+W+R+[wo]
            lrs=([L*M*a/D*eta]*L)+([L*M*a*D*eta]*L)+([L*a*math.sqrt(M)/D*eta]*L)+[eta/D]
            ld=[None]*L
            def fwd():
                h=torch.einsum('pk,skd->spd',X,We)
                for l in range(L):
                    gt=torch.sigmoid(torch.einsum('spd,sde->spe',h,R[l]))
                    with torch.no_grad():
                        q=gt+b[l].unsqueeze(1)
                        m=torch.zeros_like(q).scatter_(-1,q.topk(a,-1).indices,1.0); ld[l]=m.mean(1)
                    z=torch.einsum('spd,sedm->sepm',h,U[l])
                    h=h+cL*((gt*m).permute(0,2,1).unsqueeze(-1)*torch.einsum('sepm,semd->sepd',torch.tanh(z),W[l])).sum(1)/a
                return h
            for p in ps: p.requires_grad_(True)
            for _ in range(steps):
                f=torch.einsum('spd,sd->sp',fwd(),wo)
                gs=torch.autograd.grad((0.5*(f-Y).pow(2).mean(-1)).sum(),ps)
                with torch.no_grad():
                    for p,gr,lr in zip(ps,gs,lrs): p-=lr*gr
                    for l in range(L):
                        if ld[l] is not None: b[l]-=eb*(ld[l]-kap)
            with torch.no_grad():
                f=torch.einsum('spd,sd->sp',fwd(),wo)
                v=(0.5*(f-Y).pow(2).mean(-1)).cpu().numpy()
            v=v[np.isfinite(v)]; tot+=float(v.sum()); n+=len(v)
            for p in ps: p.requires_grad_(False)
            del P,U,W,R,b,We,wo,ps; torch.cuda.empty_cache(); done+=s
        losses.append(tot/max(n,1) if n else 1e9)
    Ls=np.array(losses); i=int(np.argmin(Ls)); lg=np.log10(grid)
    if 0<i<len(Ls)-1:
        d=Ls[i-1]-2*Ls[i]+Ls[i+1]
        v=float(lg[i]-0.5*(Ls[i+1]-Ls[i-1])/d*(lg[i]-lg[i-1])) if abs(d)>1e-12 else float(lg[i])
    else: v=float(lg[i])
    return v,(i==0 or i==len(Ls)-1),losses
try:
    RES['E2_transfer']={}
    base=dict(L=8,D=64,M=128,E=16,kap=0.25)
    dials=[('depth L','L',[4,8,16,32,64]),('active a','E',[8,16,32,64,128]),
           ('expert width M','M',[32,64,128,256,512]),('embedding D','D',[16,32,64,128,256]),
           ('expert count E (a=4)','Efix',[16,32,64,128,256])]
    for nm,k,vals in dials:
        o=[];ed=[]
        for v in vals:
            cfg=dict(base)
            if k=='Efix': cfg['E']=v; cfg['kap']=4.0/v
            else: cfg[k]=v
            try:
                lr,edge,_=opt_lr(**cfg); o.append(lr); ed.append(edge)
                log('E2 %-22s %-6s lr* %+.3f %s'%(nm,v,lr,'EDGE' if edge else ''))
            except Exception as e:
                log('E2 fail %s=%s : %r'%(nm,v,e)); o.append(None); ed.append(True)
            torch.cuda.empty_cache(); save()
        ok=[x for x in o if x is not None]
        RES['E2_transfer'][nm]={'vals':vals,'lrstar':o,'edge':ed,
            'drift':(max(ok)-min(ok)) if len(ok)>1 else None,
            'tail':(max(ok[-3:])-min(ok[-3:])) if len(ok)>=3 else None}
        log('E2 %-22s drift %.3f  tail %.3f'%(nm,RES['E2_transfer'][nm]['drift'] or -1,RES['E2_transfer'][nm]['tail'] or -1))
        save()
except Exception as e:
    log('E2 FAIL %r'%e); RES['E2_error']=repr(e); traceback.print_exc(); save()

log('ALL DONE'); RES['done']=True; save()
