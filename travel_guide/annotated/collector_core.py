# Выдержки методов GenerationConfidenceCollector, без загрузки модели.
# Оригинал: solutions/uncertainty-profiling/uncertainty_profile/extraction.py
# Это код для чтения, не самостоятельный collector. См. ../route/03-uncertainty.md.
from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Это выдержки для чтения. Запускай travel_guide/labs/03_uncertainty.py")


class GenerationConfidenceCollector:
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Reduce the current generation scores and retain them for one step.

        Args:
            input_ids: Input and generated token IDs available before selection.
            scores: Next-token vocabulary scores for each response in the batch.

        Returns:
            Unmodified ``scores`` so greedy generation semantics are preserved.
        """

        # GUIDE: scores относятся к НОВОМУ выбору; последний input_id —
        # результат ПРЕДЫДУЩЕГО выбора. Поэтому нужен pending прошлого шага.
        if self.pending_log_probs is not None:
            previous_token_ids = input_ids[:, -1].to(self.pending_log_probs.device)
            self.selected_log_probs.append(
                self.pending_log_probs
                .gather(-1, previous_token_ids.unsqueeze(-1))
                .squeeze(-1)
                .detach()
            )

        # GUIDE: нормировка [B,V] по словарю. Выбранные log-probabilities
        # имеют форму [B]; они не равны всему распределению [B,V].
        log_probs = functional.log_softmax(scores.float(), dim=-1)
        probabilities = log_probs.exp()
        entropy_terms = torch.nan_to_num(
            probabilities * log_probs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        top2_log_probs, top2_token_ids = log_probs.topk(k=2, dim=-1)
        top2_probs = top2_log_probs.exp()

        # GUIDE: 0*log(0) даёт NaN в арифметике, хотя предельный вклад равен 0.
        # nan_to_num выше обрабатывает такие значения. Итоги ниже: [B].
        self.pending_log_probs = log_probs.detach()
        self.entropy.append((-entropy_terms.sum(dim=-1)).detach())
        self.top1_probs.append(top2_probs[:, 0].detach())
        self.top2_margins.append((top2_probs[:, 0] - top2_probs[:, 1]).detach())
        self.top1_token_ids.append(top2_token_ids[:, 0].detach())
        # GUIDE: наблюдаем распределение, но не изменяем его.
        return scores

    def compute_features(
        self,
        *,
        sequences: torch.Tensor,
        prompt_length: int,
        pad_token_id: int | None,
        eos_token_ids: set[int],
        config: GenerationConfidenceConfig,
    ) -> tuple[list[ConfidenceMetrics], torch.Tensor]:
        """Finalize selected-token values and compute batch feature rows.

        Args:
            sequences: Complete prompt and generated token sequences.
            prompt_length: Padded prompt length shared by the batch.
            pad_token_id: Token ID excluded as generation padding.
            eos_token_ids: EOS token IDs excluded from feature summaries.
            config: Uncertainty metric configuration.

        Returns:
            Per-response scalar metrics and generated token IDs on CPU.

        Raises:
            RuntimeError: If generation produced no statistics or step alignment
                is inconsistent.
        """

        if not self.top1_probs or self.pending_log_probs is None:
            raise RuntimeError("generation produced no token statistics")

        num_steps = len(self.top1_probs)
        # GUIDE: следующего __call__ уже нет. Обрабатываем последний выбор
        # из sequences [B,T+G], где T = prompt_length с учётом padding.
        final_token_ids = sequences[:, prompt_length + num_steps - 1].to(
            self.pending_log_probs.device
        )
        self.selected_log_probs.append(
            self.pending_log_probs
            .gather(-1, final_token_ids.unsqueeze(-1))
            .squeeze(-1)
            .detach()
        )
        self.pending_log_probs = None
        if len(self.selected_log_probs) != num_steps:
            raise RuntimeError("generated token statistics are misaligned")

        generated_token_ids = sequences[
            :, prompt_length : prompt_length + num_steps
        ].detach().cpu()
        # GUIDE: G элементов формы [B] складываются в [B,G], не в [G,B].
        selected_log_probs = torch.stack(self.selected_log_probs, dim=1).float().cpu()
        selected_probs = selected_log_probs.exp()
        entropy = torch.stack(self.entropy, dim=1).float().cpu()
        top1_probs = torch.stack(self.top1_probs, dim=1).float().cpu()
        top2_margins = torch.stack(self.top2_margins, dim=1).float().cpu()
        top1_token_ids = torch.stack(self.top1_token_ids, dim=1).cpu()
        selected_is_top1 = generated_token_ids == top1_token_ids
        valid_mask = build_valid_token_mask(
            generated_token_ids,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
        )

        rows: list[ConfidenceMetrics] = []
        for batch_index in range(generated_token_ids.shape[0]):
            mask = valid_mask[batch_index]

            def masked_numpy(values: torch.Tensor) -> np.ndarray:
                """Select valid values for the current response as an array."""

                return values[batch_index][mask].numpy()

            # GUIDE: у строки осталось n_i токенов. Каждый вход metrics — [n_i].
            # NumPy-функция возвращает скаляры; n_i может различаться по строкам.
            rows.append(
                compute_generation_confidence_metrics(
                    log_probs=masked_numpy(selected_log_probs),
                    probs=masked_numpy(selected_probs),
                    entropy=masked_numpy(entropy),
                    top1_probs=masked_numpy(top1_probs),
                    top2_margins=masked_numpy(top2_margins),
                    selected_is_top1=masked_numpy(selected_is_top1),
                    min_k_fraction=config.min_k_fraction,
                    high_conf_threshold=config.high_conf_threshold,
                    low_conf_threshold=config.low_conf_threshold,
                )
            )
        return rows, generated_token_ids
