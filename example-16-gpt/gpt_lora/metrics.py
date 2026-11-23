import time
import statistics
import torch

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml = None
    _NVML = None


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def now(device):
    """Timestamp fiable: sincroniza el dispositivo antes de leer el reloj."""
    sync(device)
    return time.perf_counter()


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[k]


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device):
    """Pico de memoria: VRAM en CUDA, RSS del proceso en CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    if psutil:
        return psutil.Process().memory_info().rss / 1024**2
    return float("nan")


def gpu_stats():
    """Utilización, potencia (W) y temperatura. Solo NVIDIA con pynvml."""
    if _NVML is None:
        return {}
    util = pynvml.nvmlDeviceGetUtilizationRates(_NVML)
    return {
        "gpu_util_pct": util.gpu,
        "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(_NVML) / 1000,
        "gpu_temp_c": pynvml.nvmlDeviceGetTemperature(_NVML, pynvml.NVML_TEMPERATURE_GPU),
    }


def cpu_stats():
    if psutil is None:
        return {}
    return {
        "cpu_util_pct": psutil.cpu_percent(interval=None),
        "ram_rss_mb": psutil.Process().memory_info().rss / 1024**2,
        "torch_threads": torch.get_num_threads(),
    }


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


class TrainThroughput:
    """Acumula tiempo de step y tokens. Excluye warm-up y evaluación."""

    def __init__(self, device, pad_token_id=50256, warmup_steps=3):
        self.device = device
        self.pad = pad_token_id
        self.warmup = warmup_steps
        self.step_times, self.wait_times = [], []
        self.tokens_total = self.tokens_real = self.examples = 0
        self._n = 0
        self._last_end = None

    def start_step(self):
        t = now(self.device)
        if self._last_end is not None and self._n > self.warmup:
            # tiempo esperando al dataloader (y evaluación si la hubo)
            self.wait_times.append(t - self._last_end)
        self._t0 = t

    def end_step(self, input_batch):
        t = now(self.device)
        self._last_end = t
        self._n += 1
        if self._n <= self.warmup:
            return None
        dt = t - self._t0
        n_tok = input_batch.numel()
        n_real = int((input_batch != self.pad).sum())
        self.step_times.append(dt)
        self.tokens_total += n_tok
        self.tokens_real += n_real
        self.examples += input_batch.shape[0]
        return {"step_time": dt, "tok_per_s": n_tok / dt, "real_tok_per_s": n_real / dt}

    def pause(self):
        """Llamar tras una evaluación para que no cuente como espera del dataloader."""
        self._last_end = None

    def summary(self):
        total = sum(self.step_times)
        if total == 0:
            return {}
        return {
            "steps_measured": len(self.step_times),
            "tok_per_s_avg": self.tokens_total / total,
            "real_tok_per_s_avg": self.tokens_real / total,
            "examples_per_s": self.examples / total,
            "step_time_mean_s": statistics.mean(self.step_times),
            "step_time_p95_s": percentile(self.step_times, 95),
            "dataloader_wait_mean_s": statistics.mean(self.wait_times) if self.wait_times else 0.0,
            "padding_pct": 100 * (1 - self.tokens_real / self.tokens_total),
        }


@torch.inference_mode()
def generate_timed(model, idx, max_new_tokens, context_size, device, eos_id=None):
    """
    Greedy sin KV-cache (igual que generate_text_simple) pero midiendo cada token.
    Devuelve (idx, métricas). Asume batch=1 para TTFT/ITL.
    """
    prompt_len = idx.shape[1]
    stamps = []
    t_start = now(device)

    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        stamps.append(now(device))
        if eos_id is not None and idx_next.item() == eos_id:
            break

    n = len(stamps)
    ttft = stamps[0] - t_start
    e2e = stamps[-1] - t_start
    itl = [b - a for a, b in zip(stamps, stamps[1:])]
    tpot = statistics.mean(itl) if itl else float("nan")

    return idx, {
        "prompt_tokens": prompt_len,
        "new_tokens": n,
        "ttft_s": ttft,
        "e2e_s": e2e,
        "tpot_s": tpot,
        "itl_p95_s": percentile(itl, 95),
        "prefill_tok_per_s": prompt_len / ttft,          # aprox: incluye 1er decode
        "decode_tok_per_s": (1 / tpot) if itl else float("nan"),
        "total_tok_per_s": n / e2e,
    }