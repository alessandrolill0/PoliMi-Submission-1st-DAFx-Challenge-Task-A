import torch
import torchaudio
import os
import numpy as np
import torch.nn.functional as F


def load_challenge_npz(filepath, device='cpu', dtype=torch.float64):
    """Load impulse response from NPZ file."""
    data = np.load(filepath)
    waveform_np = data['ir'] 
    waveform = torch.from_numpy(waveform_np).to(device=device, dtype=dtype)
    return waveform.squeeze()


def load_target_audio(filepath: str, target_sr: int = 44100, device: torch.device = torch.device('cpu'),
                      dtype: torch.dtype = torch.float64, normalize: bool = True) -> torch.Tensor:
    """Load and resample audio file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    waveform, sr = torchaudio.load(filepath, backend='soundfile')
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    waveform = waveform.squeeze(0)

    if normalize:
        peak = torch.max(torch.abs(waveform)) + 1e-8
        waveform = waveform / peak

    return waveform.to(device=device, dtype=dtype)

def atanh_safe(u):
    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    return 0.5 * np.log(u / (1.0 - u))


def inverse_map_range_linear(y, min_v, max_v):
    norm_y = (y - min_v) / (max_v - min_v)
    x_raw = atanh_safe(norm_y)
    return float(x_raw)


def inverse_map_range_log(y, min_v, max_v):
    y = np.clip(y, 1e-15, np.inf)
    min_v = np.clip(min_v, 1e-15, np.inf)
    max_v = np.clip(max_v, 1e-15, np.inf)
    
    log_y = np.log(y)
    log_min = np.log(min_v)
    log_max = np.log(max_v)
    
    norm_y = (log_y - log_min) / (log_max - log_min)
    x_raw = atanh_safe(norm_y)
    
    return float(x_raw)

def to_norm(x, min_v, max_v, scale=1.0):
    """Legacy: scale tanh output to [min_v, max_v]."""
    return min_v + (max_v - min_v) * ((torch.tanh(x * scale) + 1.0) / 2.0)


def map_range_log(x, min_v, max_v, dtype=torch.float32, device='cpu', weight=1.0, eps=1e-10):
    min_v = torch.as_tensor(min_v, dtype=dtype, device=device)
    max_v = torch.as_tensor(max_v, dtype=dtype, device=device)
    
    min_v = torch.clamp(min_v, min=eps)  # ✅ Input validation
    max_v = torch.clamp(max_v, min=eps)
    
    norm_x = (torch.tanh(x * weight) + 1.0) / 2.0  # [0, 1]
    
    log_min = torch.log(min_v)
    log_max = torch.log(max_v)
    val_log = log_min + norm_x * (log_max - log_min)
    
    result = torch.exp(val_log)
    
    return result


def map_range_linear(x, min_v, max_v, dtype=torch.float32, device='cpu', weight=1.0):
    min_v = torch.as_tensor(min_v, dtype=dtype, device=device)
    max_v = torch.as_tensor(max_v, dtype=dtype, device=device)

    norm_x = (torch.tanh(x) + 1.0) / 2.0  # [0, 1]
    result = min_v + norm_x * (max_v - min_v)
    
    return result

def map_softplus_linear(x, min_v, max_v, dtype=torch.float32, device='cpu', weight=1.0):
    min_v = torch.as_tensor(min_v, dtype=dtype, device=device)
    max_v = torch.as_tensor(max_v, dtype=dtype, device=device)

    norm_x = F.softplus(x * weight)
    
    result = min_v + norm_x * (max_v - min_v)
    
    return result


def map_softplus_log(x, min_v, max_v, dtype=torch.float32, device='cpu', weight=1.0, eps=1e-10):
    min_v = torch.as_tensor(min_v, dtype=dtype, device=device)
    max_v = torch.as_tensor(max_v, dtype=dtype, device=device)
    
    min_v = torch.clamp(min_v, min=eps) 
    max_v = torch.clamp(max_v, min=eps)
    
    norm_x = torch.clamp(F.softplus(x * weight), max=1.0)
    
    log_min = torch.log(min_v)
    log_max = torch.log(max_v)
    
    val_log = log_min + norm_x * (log_max - log_min)
    
    result = torch.exp(val_log)
    
    return result


def inverse_softplus_safe(u):
    u = np.clip(u, 1e-15, np.inf)
    if u > 20.0:
        return float(u)
    
    return float(np.log(np.expm1(u)))

def inverse_map_softplus_log(y, min_v, max_v, weight=1.0):
    y = np.clip(y, 1e-15, np.inf)
    min_v = np.clip(min_v, 1e-15, np.inf)
    max_v = np.clip(max_v, 1e-15, np.inf)
    
    log_y = np.log(y)
    log_min = np.log(min_v)
    log_max = np.log(max_v)
    
    norm_y = (log_y - log_min) / (log_max - log_min)
  
    norm_y = np.clip(norm_y, 1e-15, 1.0)
    
    x_raw = inverse_softplus_safe(norm_y) / weight
    
    return float(x_raw)

def inverse_map_softplus_linear(y, min_v, max_v):
    norm_y = (y - min_v) / (max_v - min_v)
    x_raw = inverse_softplus_safe(norm_y)
    return float(x_raw)


def inverse_map_sigm_linear(y, min_v, max_v, scale=1.0):
    norm_y = (y - min_v) / (max_v - min_v)
    norm_y = np.clip(norm_y, 1e-6, 1.0 - 1e-6)
    x_raw = np.log(norm_y / (1.0 - norm_y)) / scale
    return float(x_raw)


def inverse_map_sigm_log(y, min_v, max_v, scale=1.0, temperature=1.0):
    log_y = np.log10(y)
    log_min = np.log10(min_v)
    log_max = np.log10(max_v)

    norm_y = (log_y - log_min) / (log_max - log_min)
    norm_y = np.clip(norm_y, 1e-6, 1.0 - 1e-6)

    x_raw = np.log(norm_y / (1.0 - norm_y)) / scale
    return float(x_raw)


def map_sigm_linear(x, min_v, max_v, dtype=torch.float32, device='cpu', weight=1.0):
    norm_x = torch.sigmoid(x)
    res = min_v + norm_x * (max_v - min_v)
    return res
   

def map_sigm_log(x, min_v, max_v, dtype=torch.float32, device='cpu', weight=1.0, eps=1e-10):
    min_v = torch.as_tensor(min_v, dtype=dtype, device=device)
    max_v = torch.as_tensor(max_v, dtype=dtype, device=device)

    norm_x = torch.sigmoid(x * weight)
    log_min = torch.log10(min_v)
    log_max = torch.log10(max_v)
    result = 10.0 ** (log_min + norm_x * (log_max - log_min))
    return result
