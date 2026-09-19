"""Paper's interior stationarity equation, independently of the alpha grid."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq


def moments(s, beta, k, n):
    r = 1/(s+k)
    b11 = np.sum(beta**2*s*r)
    b12 = np.sum(beta**2*s*r*r)
    b23 = np.sum(beta**2*s*s*r**3)
    d12 = np.sum(s*r*r)
    d23 = np.sum(s*s*r**3)
    d2 = np.sum((s*r)**2)
    h = (n-d2)*k*b23/d23-k*k*b12
    return b11, b12, b23, d12, d23, d2, h


def stationary_table(s, beta, n, sigmas):
    """Find feasible local minima; mark missing/non-global interior optima.

    n is the final fit size (paper convention), not the n-1 LOOCV proxy.
    """
    S = float(s@beta**2)
    floor = (brentq(lambda k: np.sum(s/(s+k))-n, 0, s.sum()/n,
                    xtol=1e-15) if len(s) > n else 0.)
    grid = np.geomspace(max(floor*(1+1e-10), s.max()*1e-14), s.max()*1e8, 1400)
    h = np.array([moments(s, beta, k, n)[-1] for k in grid])
    rows = []
    for sigma in sigmas:
        f = h-sigma**2
        roots = []
        for i in np.flatnonzero((f[:-1] < 0) & (f[1:] >= 0)):
            k = brentq(lambda k: moments(s, beta, k, n)[-1]-sigma**2,
                       grid[i], grid[i+1], xtol=1e-14, rtol=1e-13)
            b11, b12, b23, d12, d23, d2, _ = moments(s, beta, k, n)
            risk = n/(n-d2)*(k*k*b12+sigma**2)-sigma**2
            roots.append((risk, k))
        row = dict(sigma=sigma, ratio=sigma**2/S, n_stationary=n,
                   kappa_floor=floor, interior_minima=len(roots), status='no_interior_minimum',
                   kappa=np.nan, lam=np.nan, z_acc=np.nan, E_acc=np.nan,
                   R2_acc=np.nan, slope_acc=np.nan, E_gen=np.nan)
        if roots:
            risk, k = min(roots)
            b11, b12, b23, d12, d23, d2, hh = moments(s, beta, k, n)
            z = k*d12/b11*(b23/d23-b12/d12)
            muD = np.sum(beta**2*(s/(s+k))**2)+d12/(n-d2)*(k*k*b12+sigma**2)
            np.testing.assert_allclose(z, muD/b11-1, atol=1e-10, rtol=1e-8)
            _, b12lo, _, _, _, d2lo, _ = moments(s, beta, grid[0], n)
            boundary_risk = n/(n-d2lo)*(grid[0]**2*b12lo+sigma**2)-sigma**2
            row.update(status='interior_minimum', kappa=k,
                       lam=k*(1-np.sum(s/(s+k))/n), z_acc=z,
                       E_acc=(z/(1+z))**2, R2_acc=1-z*z,
                       slope_acc=1/(1+z), E_gen=risk/S,
                       boundary_E_gen=boundary_risk/S,
                       better_than_boundaries=bool(risk <= min(boundary_risk, S)),
                       stationarity_residual=hh-sigma**2)
        rows.append(row)
    return pd.DataFrame(rows)


def extrapolate_at_boundary(table, s, beta):
    """Requested direct substitution, deliberately NOT a stationary prediction."""
    result = table.copy()
    result['extrapolated'] = result.status == 'no_interior_minimum'
    for idx in result.index[result.extrapolated]:
        k = result.loc[idx, 'kappa_floor']
        n = result.loc[idx, 'n_stationary']
        b11, b12, b23, d12, d23, _, _ = moments(s, beta, k, n)
        z = k*d12/b11*(b23/d23-b12/d12)
        result.loc[idx, ['kappa', 'lam', 'z_acc', 'E_acc', 'R2_acc', 'slope_acc']] = [
            k, 0., z, (z/(1+z))**2, 1-z*z, 1/(1+z)]
        result.loc[idx, 'status'] = 'boundary_formula_extrapolation'
    return result


if __name__ == '__main__':
    import time
    from scripts.pixel_ridge_notebook_utils import load_problem
    root = Path(__file__).resolve().parents[1]
    out = root/'notebooks/outputs/pixel_ridge/vanhateren_selection_estimation'
    table = pd.read_csv(out/'comparison_extended.csv')
    s, beta = load_problem(root, 'vanhateren')
    start = time.perf_counter()
    result = stationary_table(s, beta, 1000, np.sort(table.sigma.unique()))
    result.to_csv(out/'stationary_cv.csv', index=False)
    extrapolate_at_boundary(result, s, beta).to_csv(out/'stationary_cv_extrapolated.csv', index=False)
    print(result[['sigma','status','kappa','lam','z_acc']].to_string(index=False))
    print(f'Elapsed {time.perf_counter()-start:.2f}s; cached stationary_cv.csv')
