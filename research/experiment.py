"""Читать сверху вниз: признаки → обучение головы → честное сравнение.
CLI/файлы/кеш находятся в _support.py. python -m research.experiment --help
"""
import torch
from torch.nn import functional as F
from research.features import extract_features
from research.predictor import build_predictor, normalize, predict
from research import _support as io


def run(args, record) -> None:
    train, valid = io.load_splits(args.train, args.validation, args.model)
    # X не получает меток. Кеш зависит от текстов, модели, токенизации и кода извлечения.
    rows = train + valid
    x, feature_spec = io.features(args, [r["problem"] for r in rows], extract_features)
    y = torch.tensor([r["is_robust"] for r in rows], dtype=torch.float32)
    n = len(train)
    x_train, x_valid = x[:n], x[n:]
    y_train, y_valid = y[:n], y[n:]

    # Для каждой координаты: центр и масштаб оцениваем без validation.
    flat = x_train.flatten(start_dim=1)
    mean = flat.mean(dim=0)
    scale = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
    z_train = normalize(x_train, mean, scale)
    torch.manual_seed(args.seed)
    head = build_predictor(z_train.shape[1], args.width)  # голова обучается на CPU
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    head.train()
    for _ in range(args.epochs):
        io.check_deadline(args)
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for indices in order.split(args.batch_size):
            io.check_deadline(args)
            logits = head(z_train[indices]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_train[indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append(total / n)

    # Validation не выбирает эпоху, признаки или порог внутри этого запуска.
    p_train = predict(head, x_train, mean, scale)
    p_valid = predict(head, x_valid, mean, scale)
    decision = p_valid >= args.threshold
    correct = decision == y_valid.bool()
    majority = bool(y_train.mean() >= 0.5)  # baseline выбирается по TRAIN
    # Отдельно равный вес исходным задачам, а не числу их перефразов.
    families = sorted({r["family_id"] for r in valid})
    group_accuracy = [float(correct[[i for i, r in enumerate(valid) if r["family_id"] == g]].float().mean())
                      for g in families]
    record.update(feature_spec=feature_spec, shape=list(x.shape), train_loss=history,
                  train_rows=n, validation_rows=len(valid), validation_families=len(families),
                  train_accuracy=float(((p_train >= args.threshold) == y_train.bool()).float().mean()),
                  accuracy=float(correct.float().mean()), family_accuracy=sum(group_accuracy) / len(families),
                  majority_accuracy=float((y_valid.bool() == majority).float().mean()))
    io.save_outputs(args, head, mean, scale, feature_spec, valid, p_valid)


if __name__ == "__main__":
    io.main(run)
