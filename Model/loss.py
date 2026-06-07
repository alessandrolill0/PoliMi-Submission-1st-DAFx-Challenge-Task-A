import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):

    def __init__(self, mse_weight=0.0, stft_weight=1.0,
                 energy_weight=0.0,
                 cutoff_hz=200.0, sr=44100,
                 fft_sizes=[64, 256, 1024, 4096]):
        super().__init__()

        self.mse_weight = mse_weight
        self.stft_weight = stft_weight
        self.energy_weight = energy_weight
        self.sr = sr
        self.fft_sizes = fft_sizes

        self.windows = nn.ParameterDict({
            str(n): nn.Parameter(torch.hann_window(n), requires_grad=False)
            for n in fft_sizes
        })
            
        self.target_stft_cache = {}  # key: (device, n_fft), value: stft tensor
        self.cached_target_audio = None

        self.eps = 1e-9

    def precompute_target_stft(self, target_audio):
        target_audio = target_audio.squeeze()
        device = target_audio.device
        
        self.cached_target_audio = target_audio
        self.target_stft_cache.clear()
        
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            window = self.windows[str(n_fft)].to(device)
            
            target_stft = torch.stft(target_audio, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=window,
                                     return_complex=True, center=True)
            
            self.target_stft_cache[(device, n_fft)] = target_stft
    
    def forward(self, pred_audio, target_audio):
        peak = torch.max(torch.abs(target_audio)) + 1e-8
        norm_target = target_audio / peak
        norm_pred = pred_audio / peak
        pred_audio = pred_audio.squeeze()
        target_audio = target_audio.squeeze()

        device = pred_audio.device

        scale = 1e6
        pred_sc = pred_audio.squeeze() * scale
        target_sc = target_audio.squeeze() * scale

       # =========================
        # 1. TIME-DOMAIN MSE
        # =========================
        target_variance = torch.mean(target_sc**2).clamp_min(self.eps)
        
        mse_loss = F.mse_loss(pred_sc, target_sc) / target_variance

       # =========================
        # 2. GLOBAL ENERGY LOSS (Volume Assoluto)
        # =========================
        pred_total_energy = torch.sum(pred_sc**2)
        target_total_energy = torch.sum(target_sc**2)
        
        # log(A) - log(B). La scala si annulla matematicamente, la precisione si salva!
        energy_loss = F.l1_loss(
            torch.log(pred_total_energy + self.eps), 
            torch.log(target_total_energy + self.eps)
        )

        # =========================
        # 3. MULTI-SCALE STFT
        # =========================
        mss_loss = torch.tensor(0.0, device=device, dtype=pred_audio.dtype)

        if self.stft_weight > 0:
            for n_fft in self.fft_sizes:
                hop = n_fft // 4
                window = self.windows[str(n_fft)].to(device)

                pred_stft = torch.stft(pred_audio, n_fft=n_fft, hop_length=hop,
                                       win_length=n_fft, window=window,
                                       return_complex=True, center=True)

                target_stft = self.target_stft_cache[(device, n_fft)]

                pred_mag   = torch.abs(pred_stft).clamp_min(self.eps)
                target_mag = torch.abs(target_stft).clamp_min(self.eps)

                sc       = torch.norm(target_mag - pred_mag) / torch.norm(target_mag)
                log_loss = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))

                mss_loss = mss_loss + (sc + log_loss)

            mss_loss = mss_loss / len(self.fft_sizes)

        # =========================
        # FINAL
        # =========================
        total_loss = (
            self.mse_weight * mse_loss +
            self.stft_weight * mss_loss +
            self.energy_weight * energy_loss
        )

        return total_loss