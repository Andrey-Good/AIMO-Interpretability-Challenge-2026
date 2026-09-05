# Тензоры и структуры: шпаргалка

[Начало](../README.md). Формулы с объяснениями находятся в главах [probe](../route/02-probe.md) и [uncertainty](../route/03-uncertainty.md); здесь — быстрый поиск формы и владельца объекта.

## Общие обозначения

`N` — все задачи в одном вызове; `B` — размер фактического батча LLM; `T` — длина входа после padding; `G` — число сгенерированных шагов; `D` — ширина hidden state; `V` — словарь; `L` — число блоков; `S` — пробы одной группы. Все численные примеры guide искусственные. Реальные размеры печатает [необязательный пример](../labs/04_cached_model.py).

## Representation probe

`problems: list[str]`, длина `N` → цикл по строкам в `predict_robustness`. Батч LLM здесь всегда состоит из одной строки, а не из всех `N` задач.

`input_ids: torch.Tensor [1,T]`, обычно `torch.int64`, создаёт tokenizer. `attention_mask: [1,T]` обозначает допустимые позиции; это не матрица весов attention `[B,heads,T,T]`.

`output.hidden_states: tuple` длины обычно `L+1`; каждый элемент `[1,T,D]`, dtype модели, устройство модели. `_encode_problem` выбирает один индекс tuple, затем `[0,-1,:]` и возвращает **`np.ndarray [D], float32, CPU`**.

`weights: [S,D]`, `bias: [S]`, `threshold: [S]` создаются не forward-проходом, а загружаются из артефакта. `weights @ vector` — `[S]`; margins всех групп объединяются в плоский список. Выход `mean_ensemble_margin` — Python `float`, выход `_predict_problem` — Python `bool`.

**При переходе к батчам:** с left padding последняя колонка соответствует последней позиции каждого промпта. С right padding последняя колонка короткой строки может быть pad. Не заменяй цикл на `H[:, -1, :]` механически без проверки маски. Среднее по токенам также должно исключать pad. У разных моделей терминальные состояния могут включать нормализацию — слой и точку чтения фиксируй явно.

## Uncertainty profiling

`encoded['input_ids']: [B,T]`, целые ID; tokenizer использует left padding. `prompt_length` берётся как `shape[1]` **всего батча**, включая padding, чтобы правильно отделить продолжение.

`collector.__call__(input_ids, scores)` получает IDs уже имеющегося префикса `[B,T+t]` и scores следующего токена `[B,V]`. `log_softmax(...float())` даёт log-probabilities float32 `[B,V]`. `pending_log_probs` хранит только один такой шаг.

Списки `selected_log_probs`, `entropy`, `top1_probs`, `top2_margins` содержат векторы `[B]` по шагам. После `torch.stack(..., dim=1)` — `[B,G]`; после переноса на CPU и маски для строки — NumPy `[n_i]`, где `n_i` — число её токенов без EOS/pad. `top1_token_ids` и `generated_token_ids` имеют целый dtype, `valid_mask` — bool.

`outputs.sequences: [B,T+G]` → срез `[:, prompt_length:prompt_length+num_steps]` → generated IDs `[B,G]`. Не смешивай срез continuation с последним токеном промпта из probe: это разные признаки.

`feature_rows: list[dict]` → `pd.DataFrame(..., columns=artifact.feature_names): [N,14]` → `np.asarray(estimator.predict(...)): [N]` → `list[bool]` длины `N`. Столбцы обязаны сохранять порядок. Строки обязаны сохранять соответствие входным задачам.

## Что лежит в артефактах

Схемы ниже установлены **по загрузчикам**, а не путём исполнения бинарников.

`ProbeArtifact` содержит `model_id`, `system_prompt`, `kind`, `data`. Для `.pkl`: `schema_version=2`, `artifact_type='all_folds_layers_seed_probe_ensemble'`, `best_layer_index`, необязательный `recommended_strategy`, непустой `groups`. Внутри `group['probes'][layer]` лежат `weights`, `bias`, `threshold`. Ключ слоя допускается числом или строкой. Группы без выбранного слоя пропускаются; если подходящих проб не осталось, возникает ошибка. Для legacy `.npz` сохраняется одна проба: `layer_index`, `weights:[D]`, скалярные `bias` и `threshold`.

`UncertaintyArtifact`: `schema_version=1`, тип `uncertainty_regressor`, `model_id`, `feature_names`, `generation_config`, обученный `estimator`, `decision_threshold`. Ещё есть `selected_params`, `cv_results`, `training_provenance`, `library_versions`. Это объясняет, почему новые defaults не перенастраивают старый estimator. Загружается файл с нормализованным именем checkpoint, и проверяется соответствие его `model_id` запросу.

`pickle.load` и `joblib.load` могут исполнять код из файла: не используй их как безобидный просмотрщик неизвестных бинарников. Проверка словаря после загрузки проверяет схему, но не делает саму десериализацию безопасной.
