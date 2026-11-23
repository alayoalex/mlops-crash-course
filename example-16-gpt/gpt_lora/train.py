from utils import calc_accuracy_loader, calc_loss_batch, evaluate_model
from metrics import TrainThroughput, reset_peak_memory, peak_memory_mb, gpu_stats, cpu_stats
import mlflow


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, pad_token_id=50256, log_fn=lambda d, step: mlflow.log_metrics(d, step=step)):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    tp = TrainThroughput(device, pad_token_id=pad_token_id, warmup_steps=3)
    reset_peak_memory(device)

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            tp.start_step()

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            m = tp.end_step(input_batch)   # sincroniza y mide (None durante warm-up)
            examples_seen += input_batch.shape[0]
            global_step += 1

            if m and log_fn:               # p. ej. log_fn = mlflow.log_metrics
                log_fn({**m, "loss": loss.item()}, step=global_step)

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                tp.pause()                 # la evaluación no cuenta como espera
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                extra = ""
                if m:
                    extra = f" | {m['tok_per_s']:.0f} tok/s ({m['real_tok_per_s']:.0f} reales)"
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}{extra}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        tp.pause()
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    summary = tp.summary()
    summary["peak_mem_mb"] = peak_memory_mb(device)
    summary.update(gpu_stats())
    summary.update(cpu_stats())
    print("\n=== Training throughput ===")
    for k, v in summary.items():
        print(f"{k:>24}: {v:.3f}" if isinstance(v, float) else f"{k:>24}: {v}")

    return train_losses, val_losses, train_accs, val_accs, examples_seen, summary
