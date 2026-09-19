import numpy as np
import torch
from scripts.vanhateren_landscape import make_coefficients, evaluate


def test_polynomial_matches_direct_ridge_and_explicit_loo():
    g=torch.Generator().manual_seed(7)
    X=torch.randn(12,8,generator=g,dtype=torch.float64); X-=X.mean(0)
    pop=torch.randn(30,8,generator=g,dtype=torch.float64); pop-=pop.mean(0)
    b=torch.randn(8,generator=g,dtype=torch.float64)
    eps=torch.randn(12,generator=g,dtype=torch.float64); eps-=eps.mean()
    y=X@b; target=pop@b; S=float(target.square().mean())
    alphas=np.array([.02,1.,50.])
    c,_=make_coefficients(X,y,eps,pop,target,b,alphas)
    for sigma in [0.,.3,3.]:
        vals=evaluate(c,sigma,S,float(b@b))
        for j,alpha in enumerate(alphas):
            response=y+sigma*eps
            w=torch.linalg.solve(X.T@X+alpha*torch.eye(8),X.T@response)
            p=pop@w; N=float(b@w); D=float(w@w)
            eg=float((p-target).square().mean())/S
            expected=dict(E_gen=eg,E_acc=(1-N/D)**2,R2_gen=1-eg,
                R2_acc=1-(D/N-1)**2,slope_gen=float(target@p/(p@p)),
                slope_acc=N/D,weight_error=float((w-b).square().sum()))
            for name in vals: np.testing.assert_allclose(vals[name][j],expected[name],rtol=1e-6,atol=1e-9)
            residuals=[]
            for i in range(len(X)):
                keep=torch.arange(len(X))!=i
                xx=X[keep]; yy=response[keep]
                xm=xx.mean(0); ym=yy.mean(); xx=xx-xm; yy=yy-ym
                ww=torch.linalg.solve(xx.T@xx+alpha*torch.eye(8),xx.T@yy)
                residuals.append(float((response[i]-ym-(X[i]-xm)@ww)**2))
            risk=c['CV0'][j]+2*sigma*c['CV1'][j]+sigma**2*c['CV2'][j]
            np.testing.assert_allclose(risk,np.mean(residuals),rtol=1e-6)
