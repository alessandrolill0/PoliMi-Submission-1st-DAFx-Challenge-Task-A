import numpy as np
from scipy.stats import qmc
from utils import (
    inverse_map_sigm_linear,
    inverse_map_sigm_log,
)

MU_BOUNDS      = (2.43,   106.15)
D_BOUNDS       = (0.2805, 201.188)
T0_BOUNDS      = (9.4e-5, 411.52)
LY_BOUNDS      = (1.1,    4.0)
XO_FRAC_BOUNDS = (0.51,   1.0)   
YO_FRAC_BOUNDS = (0.51,   1.0)   

T0_WEIGHT = 2.0 
D0_WEIGHT = 2.0 



def lhs_sample_raw_params(n_starts: int, seed: int = 42) -> list[dict]:
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    unit_samples = sampler.random(n=n_starts)

    def log_interp(u, lo, hi):
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))

    def lin_interp(u, lo, hi):
        return lo + u * (hi - lo)

    raw_params_list = []
    for u in unit_samples:
        mu         = log_interp(u[0], *MU_BOUNDS)
        D_over_mu  = log_interp(u[1], *D_BOUNDS)
        T0_over_mu = log_interp(u[2], *T0_BOUNDS)
        Ly         = lin_interp(u[3], *LY_BOUNDS)
        xo         = lin_interp(u[4], *XO_FRAC_BOUNDS)
        yo         = lin_interp(u[5], *YO_FRAC_BOUNDS) * Ly


        mu_raw         = inverse_map_sigm_log(mu,         *MU_BOUNDS, scale=1.0)
        D_over_mu_raw  = inverse_map_sigm_log(D_over_mu,  *D_BOUNDS,  scale=2.0) 
        T0_over_mu_raw = inverse_map_sigm_log(T0_over_mu, *T0_BOUNDS, scale=2.0) 
        
        Ly_raw         = inverse_map_sigm_linear(Ly, *LY_BOUNDS, scale=1.0)
        xo_raw         = inverse_map_sigm_linear(xo, *XO_FRAC_BOUNDS, scale=1.0)
        yo_raw         = inverse_map_sigm_linear(yo, 0.51 * Ly, Ly, scale=1.0)

        raw_params_list.append({
            'mu_raw':         float(mu_raw),
            'D_over_mu_raw':  float(D_over_mu_raw),
            'T0_over_mu_raw': float(T0_over_mu_raw),
            'Ly_raw':         float(Ly_raw),
            'xo_raw':         float(xo_raw),
            'yo_raw':         float(yo_raw),
        })

    return raw_params_list