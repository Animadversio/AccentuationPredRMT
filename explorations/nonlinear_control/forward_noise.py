"""Forward-only, debiased smoothed directional energy statistics."""
import numpy as np


def inner_statistics(d):
    """d: [independent outer directions, iid centers, PCs]."""
    d = np.asarray(d, dtype=np.float64)
    count = d.shape[1]
    if count < 2:
        raise ValueError('Need >=2 independent centers for an unbiased square')
    smooth = (d.sum(1)**2 - (d*d).sum(1))/(count*(count-1))
    neighborhood = (d*d).mean(1)
    return smooth, neighborhood


def step_statistics(plus, minus, baseline, tau):
    odd = ((np.asarray(plus)-np.asarray(minus))/(2*tau))**2
    even = ((np.asarray(plus)+np.asarray(minus)-2*np.asarray(baseline))/(2*tau))**2
    return odd, odd+even, even


def covariance_pseudovalues(plus, minus, baseline, tau):
    """Unbiased cloud variance and squared drift, with jackknife pseudovalues.

    Antithetic plus/minus are correlated; independent sampling units are pairs.
    Means of returned arrays are estimates; std/sqrt(R) gives jackknife SE.
    """
    plus, minus, baseline = map(lambda x: np.asarray(x,dtype=np.float64), (plus,minus,baseline))
    count=len(plus)
    if count<3:
        raise ValueError('Need at least three independent antithetic pairs')
    a=(plus-minus)/(2*tau)
    b=(plus+minus-2*baseline)/(2*tau)
    energy=a*a+b*b
    total=b.sum(0); total2=(b*b).sum(0)
    drift=(total*total-total2)/(count*(count-1))
    drift_loo=((total-b)**2-(total2-b*b))/((count-1)*(count-2))
    drift_pseudo=count*drift-(count-1)*drift_loo
    # Sample means have energy itself as their jackknife pseudovalues.
    return energy-drift_pseudo, drift_pseudo
