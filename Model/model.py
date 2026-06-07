import torch
import torchaudio
import torch.nn as nn
import numpy as np
from utils import map_range_linear, map_range_log, map_softplus_linear, map_softplus_log, map_sigm_linear,map_sigm_log
class DifferentiableModalPlate(nn.Module):

    def __init__(self, sample_rate: int = 44100, plate_params: dict = None,
                 dtype: torch.dtype = torch.float64):
        super(DifferentiableModalPlate, self).__init__()

        import math

        self.sample_rate = sample_rate
        self.k = 1.0 / sample_rate
        self.fmax = 10000.0
        self.maxOm = self.fmax * 2 * math.pi
        self.dtype = dtype

        # =========================
        # 1. FIXED PARAMETERS
        # =========================
        self.register_buffer('Lx', torch.tensor(1.0, dtype=dtype))
        self.register_buffer('tau_0', torch.tensor(6.0, dtype=dtype))
        self.register_buffer('tau_1', torch.tensor(2.0, dtype=dtype))
        self.register_buffer('loss_f1', torch.tensor(500.0, dtype=dtype))

        # =========================
        # 2. RAYLEIGH DAMPING 
        # =========================
        OmDamp1 = torch.tensor(0.0, dtype=dtype)
        OmDamp2 = 2 * torch.pi * self.loss_f1

        dOmSq = OmDamp2**2 - OmDamp1**2
        eps = torch.tensor(1e-12, dtype=dtype)
        dOmSq = torch.clamp(dOmSq, min=eps)

        log10 = math.log(10.0)
        alpha = 3 * log10 / dOmSq * (OmDamp2**2 / self.tau_0 - OmDamp1**2 / self.tau_1)
        beta  = 3 * log10 / dOmSq * (1.0 / self.tau_1 - 1.0 / self.tau_0)

        self.register_buffer('alpha', alpha.clone().detach())
        self.register_buffer('beta', beta.clone().detach())

        # =========================
        # 3. LEARNABLE PARAMETERS
        # =========================
        def init_param(name, default):
            if plate_params is None:
                return nn.Parameter(torch.tensor(default, dtype=dtype))
            else:
                if name not in plate_params:
                    raise KeyError(f"Missing parameter '{name}' in plate_params")
                return nn.Parameter(torch.tensor(plate_params[name], dtype=dtype))

        self.mu_raw         = init_param('mu_raw', 0.0)
        self.D_over_mu_raw  = init_param('D_over_mu_raw', 0.1)
        self.T0_over_mu_raw = init_param('T0_over_mu_raw', 0.5)
        self.Ly_raw         = init_param('Ly_raw', 0.0)
        self.xo_raw         = init_param('xo_raw', 0.0)
        self.yo_raw         = init_param('yo_raw', 0.0)

    def get_physical_parameters(self):
        mu = map_sigm_log(self.mu_raw, 2.43, 106.15, dtype=self.dtype, device=self.Lx.device, weight=1.0)
        D_over_mu = map_sigm_log(self.D_over_mu_raw, 0.2805, 201.188, dtype=self.dtype, device=self.Lx.device, weight=2.0)
        T0_over_mu = map_sigm_log(self.T0_over_mu_raw, 9.4e-5, 411.52, dtype=self.dtype, device=self.Lx.device, weight=2.0)
        
        Ly = map_sigm_linear(self.Ly_raw, 1.1, 4.0, dtype=self.dtype, device=self.Lx.device, weight=1.0)
        xo = map_sigm_linear(self.xo_raw, 0.51 * self.Lx, 1.0 * self.Lx, dtype=self.dtype, device=self.Lx.device, weight=1.0)
        yo = map_sigm_linear(self.yo_raw, 0.51 * Ly, 1.0 * Ly, dtype=self.dtype, device=self.Lx.device, weight=1.0)

        return mu, D_over_mu, T0_over_mu, Ly, xo, yo
    
    def solve_modal_system(self, omega: torch.Tensor, sigma: torch.Tensor, P: torch.Tensor, 
                           num_samples: int) -> torch.Tensor:
        device = omega.device
        dtype = omega.dtype
        
        n = torch.arange(num_samples, device=device, dtype=dtype).unsqueeze(0)  # [1, T]
        y = torch.zeros(num_samples, device=device, dtype=dtype)
        
        chunk_size = 1000
        for i in range(0, omega.shape[0], chunk_size):
            o_chunk = omega[i:i+chunk_size].unsqueeze(1)  # [chunk, 1]
            s_chunk = sigma[i:i+chunk_size].unsqueeze(1)  # [chunk, 1]
            p_chunk = P[i:i+chunk_size].unsqueeze(1)      # [chunk, 1]

            sin_omega_k = torch.sin(o_chunk * self.k).clamp_min(1e-12)
            n_minus_1 = torch.clamp(n - 1, min=0)

            envelope = torch.exp(-s_chunk * self.k * n_minus_1)
            oscillator = torch.sin(o_chunk * self.k * n)

            modes_chunk = (p_chunk / sin_omega_k) * envelope * oscillator
            y += torch.sum(modes_chunk, dim=0)
            
        y[0] = 0.0 
        return y

    def forward(self, duration: float = 1.0, normalize: bool = True, velCalc: bool = False) -> torch.Tensor:
        mu, D_over_mu, T0_over_mu, Ly, xo, yo = self.get_physical_parameters()

        device = self.Lx.device
        pi = torch.pi

        # =========================
        # 1. MODAL GRID 
        # =========================
        DDx = 110
        DDy = 439

        m_idx = torch.arange(1, DDx + 1, device=device, dtype=self.dtype)
        n_idx = torch.arange(1, DDy + 1, device=device, dtype=self.dtype)

        m_grid, n_grid = torch.meshgrid(m_idx, n_idx, indexing='ij')
        m_vec = m_grid.flatten()
        n_vec = n_grid.flatten()

        # =========================
        # 2. MODAL FREQUENCIES
        # =========================
        g1 = (m_vec * pi / self.Lx)**2 + (n_vec * pi / Ly)**2
        g2 = g1 * g1

        omega_sq = T0_over_mu * g1 + D_over_mu * g2
        omega = torch.sqrt(torch.clamp(omega_sq, min=0.0))

        # =========================
        # 3. TRUNCATE 
        # =========================
        valid = omega <= self.maxOm

        omega = omega[valid]
        m_vec = m_vec[valid]
        n_vec = n_vec[valid]

        # =========================
        # 4. DAMPING + COEFFICIENTS
        # =========================
        sigma = self.alpha + self.beta * omega**2
        exp_term = torch.exp(-sigma * self.k)

        frac_xi = 0.335
        frac_yi = 0.467
        frac_xo = xo / self.Lx
        frac_yo = yo / Ly

        InWeight = torch.sin(frac_xi * pi * m_vec) * torch.sin(frac_yi * pi * n_vec)
        OutWeight = torch.sin(frac_xo * pi * m_vec) * torch.sin(frac_yo * pi * n_vec)

        ms = 0.25 * mu * self.Lx * Ly
        
        P = 4.0 * OutWeight * InWeight * self.k**2 * exp_term / (ms * self.Lx * Ly)

        # =========================
        # 5. TIME INTEGRATION 
        # =========================
        num_samples = int(self.sample_rate * duration)
        
        y = self.solve_modal_system(omega, sigma, P, num_samples)

        if velCalc:
            y_prev_tensor = torch.cat((torch.zeros(1, device=device, dtype=self.dtype), y[:-1]))
            y = (y - y_prev_tensor) / self.k

        # =========================
        # 6. NORMALIZATION
        # =========================
        if normalize:
            # detach prevents peak normalization from blocking gradient flow
            peak = torch.max(torch.abs(y)).detach() + 1e-8
            y = y / peak

        return y