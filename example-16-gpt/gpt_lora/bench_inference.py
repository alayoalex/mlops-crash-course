import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import statistics
import torch
import tiktoken
from gpt.gpt import GPTModel
from metrics import (generate_timed, reset_peak_memory, peak_memory_mb,
                        gpu_stats, percentile)

CONFIG = {
    "vocab_size": 50257, "context_length": 1024, "emb_dim": 768,
    "n_heads": 12, "n_layers": 12, "drop_rate": 0.0, "qkv_bias": True,
}
PROMPT_LENS = [16, 128, 512]
NEW_TOKENS = 64
WARMUP_RUNS = 2
RUNS = 10


def bench(device_name, dtype=torch.float32):
    device = torch.device(device_name)
    torch.manual_seed(123)
    model = GPTModel(CONFIG).to(device=device, dtype=dtype).eval()
    n_threads = torch.get_num_threads()

    print(f"\n### {device_name} | dtype={dtype} | threads={n_threads}")
    print(f"{'prompt':>7} {'TTFT ms':>9} {'TPOT ms':>9} {'E2E s':>8} "
            f"{'decode t/s':>11} {'p95 ITL ms':>11} {'mem MB':>8}")

    for plen in PROMPT_LENS:
        # prompt sintético de longitud exacta
        idx = torch.randint(0, CONFIG["vocab_size"], (1, plen), device=device)
        new = min(NEW_TOKENS, CONFIG["context_length"] - plen)

        for _ in range(WARMUP_RUNS):
            generate_timed(model, idx, new, CONFIG["context_length"], device)

        reset_peak_memory(device)
        res = [generate_timed(model, idx, new, CONFIG["context_length"], device)[1]
                for _ in range(RUNS)]

        med = lambda k: statistics.median(r[k] for r in res)
        print(f"{plen:>7} {med('ttft_s')*1e3:>9.1f} {med('tpot_s')*1e3:>9.1f} "
            f"{med('e2e_s'):>8.2f} {med('decode_tok_per_s'):>11.1f} "
            f"{percentile([r['itl_p95_s'] for r in res], 95)*1e3:>11.1f} "
            f"{peak_memory_mb(device):>8.0f}")

    if device.type == "cuda":
        print("gpu:", gpu_stats())


if __name__ == "__main__":
    bench("cpu")
    if torch.cuda.is_available():
        bench("cuda")
        bench("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)