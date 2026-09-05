# Карта исходных файлов

[Начало](../README.md). Это справочник, а не порядок чтения. Внутри каждого блока пути считаются от корня репозитория; ссылки ведут к настоящим файлам. Дерево относится к исходному коммиту, указанному на главной странице.

## Ядро: понимать до самостоятельной разработки

**[solutions/always-true/solution.py](../../solutions/always-true/solution.py)** — минимальный контракт без ML: один `True` на входную строку. Полезен для отделения ошибок инфраструктуры от ошибок модели.

**[solutions/trained-probe/solution.py](../../solutions/trained-probe/solution.py)** — адаптер `are_robust → predict_robustness`.

**[solutions/trained-probe/probe_inference.py](../../solutions/trained-probe/probe_inference.py)** — весь representation inference. Читать в порядке `predict_robustness → _predict_problem → _encode_problem → mean_ensemble_margin`. `_load_model` загружает LLM; `_build_prompt` форматирует текст. `_artifact_candidates`, `_load_artifact`, `load_pickle_artifact` и `_load_npz_artifact` ищут и проверяют сохранённую пробу; `_layer_index` выбирает слой. `_single_probe_margin` применяет старый однопробный формат. `_safe_model_id` и `_scalar` — маленькие преобразователи строк и скаляров, не ML.

**[solutions/uncertainty-profiling/solution.py](../../solutions/uncertainty-profiling/solution.py)** — такой же внешний адаптер. Далее работают шесть файлов пакета:

- **[uncertainty_profile/inference.py](../../solutions/uncertainty-profiling/uncertainty_profile/inference.py)** — связывает загрузку артефакта, извлечение признаков, `estimator.predict` и порог; освобождает модель в `finally`.
- **[uncertainty_profile/extraction.py](../../solutions/uncertainty-profiling/uncertainty_profile/extraction.py)** — LLM и генерация. `load_model_and_tokenizer` готовит модель и left-padding tokenizer; `format_problem` применяет chat template. `iter_generation_feature_batches` выдаёт батчи, `extract_generation_features` объединяет их. `GenerationConfidenceCollector` собирает статистики; `build_valid_token_mask` исключает EOS/pad. `compute_batch_features_from_scores` — альтернативный путь для проверки эквивалентности, **не основной runtime**. `get_model_input_device`, `_token_id_set`, `release_model` — служебные функции устройства, ID и памяти.
- **[uncertainty_profile/metrics.py](../../solutions/uncertainty-profiling/uncertainty_profile/metrics.py)** — NumPy-формулы статистик; `_mean` задаёт поведение на пустом массиве. Здесь удобно начинать изменение готовых признаков.
- **[uncertainty_profile/config.py](../../solutions/uncertainty-profiling/uncertainty_profile/config.py)** — порядок `FEATURE_NAMES`, alias модели и `GenerationConfidenceConfig`. `validate`, `to_dict`, `from_dict` обеспечивают одинаковые настройки обучения и применения.
- **[uncertainty_profile/artifact.py](../../solutions/uncertainty-profiling/uncertainty_profile/artifact.py)** — схема артефакта. `validate_artifact_payload` проверяет содержимое, `load_artifact` ищет подходящий файл, `make_artifact_payload` и `dump_artifact` создают его. `Predictor` описывает требование иметь `.predict`, а не реализует регрессор.
- **[uncertainty_profile/__init__.py](../../solutions/uncertainty-profiling/uncertainty_profile/__init__.py)** — вход пакета; исследовательскую логику искать в перечисленных модулях.

## Обучение uncertainty: понимать при изменении метода

**[scripts/compute_uncertainty_features.py внутри solution](../../solutions/uncertainty-profiling/scripts/compute_uncertainty_features.py)** — получение обучающих строк, проверка схемы, дедупликация промптов, разделение работы на части (shards), возобновляемый кеш и возврат признаков к исходным строкам. Главные точки: `validate_source_rows`, `validate_partial_frame`, `main`. `prompt_key` — хеш текста, не обучаемый embedding; `atomic_write_parquet` — запись через временный файл.

**[scripts/train_uncertainty_regressor.py внутри solution](../../solutions/uncertainty-profiling/scripts/train_uncertainty_regressor.py)** — `load_feature_data` и `validate_feature_data` готовят строки; `build_candidate_specs` и `build_pipeline` задают модели; `make_grouped_splits` разделяет группы; `compute_oof_predictions` обучает по фолдам. `threshold_candidates`, `select_threshold`, `evaluate_predictions`, `run_model_selection` выбирают вариант. `main` переобучает победителя и проверяет сохранение/загрузку. `CandidateSpec` и `EvaluationResult` описывают настройки и результаты; файловые хеши и запись JSON — вспомогательная часть.

## Инфраструктура: понимать границы, обычно не менять

**[components/ingestion_program/ingestion.py](../../components/ingestion_program/ingestion.py)** — чтение и проверка cases, загрузка solution, группировка вызовов, валидация ответов. `run` временно добавляет путь импорта, `_run` выполняет основной цикл.

**[components/scoring_program/scoring.py](../../components/scoring_program/scoring.py)** — `load_labels`, `load_predictions`, `compute_scores`; сравнивает по ID, а не по порядку строк JSONL.

**[ingestion metadata.yaml](../../components/ingestion_program/metadata.yaml)** и **[scoring metadata.yaml](../../components/scoring_program/metadata.yaml)** — команды запуска для платформы. Это настройки оболочки, не гиперпараметры метода.

**[scripts/run_local.py](../../scripts/run_local.py)** — два процесса: ingestion и scoring. **[scripts/import_hf_dataset.py](../../scripts/import_hf_dataset.py)** — получение данных через rows API или Parquet; `convert_rows` строит cases/labels, `make_case_id` вычисляет ID. Его текущая схема problem не согласована с ingestion. **[scripts/build.py](../../scripts/build.py)** — ZIP-архивы; `solutions` ищет решения, `directory_entries` читает файлы, `archive` задаёт стабильный порядок и время ZIP. Организаторские `build_task`/`build_competition` для работы над методом не нужны.

## Артефакты, документация и окружение

**[probe_artifact.pkl](../../solutions/trained-probe/probe_artifacts/probe_artifact.pkl)** — обученные пробы и метаданные. **[DeepSeek .joblib](../../solutions/uncertainty-profiling/uncertainty_artifacts/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib)** — обученный uncertainty estimator, схема признаков, порог и метаданные. Схемы описаны [здесь](tensors.md); бинарники не продублированы и при создании guide не исполнялись.

**[Корневой README](../../README.md)** — исходные команды и описание окружения. **[README trained-probe](../../solutions/trained-probe/README.md)** и **[README uncertainty](../../solutions/uncertainty-profiling/README.md)** — дополнительная документация baseline. При конфликте с кодом смотри [заметки о несовпадениях](pitfalls.md).

**[Dockerfile.competition](../../Dockerfile.competition)** — окружение evaluator. **[pyproject.toml](../../pyproject.toml)** — локальные зависимости; `pyarrow` находится в группе разработки. **[uv.lock](../../uv.lock)** — зафиксированное разрешение зависимостей, не файл для ручного чтения. **[.gitignore](../../.gitignore)** — исключения для Git.

**[tests/test_ingestion_contract.py](../../tests/test_ingestion_contract.py)** — проверки старого и нового контрактов; часть не согласована с текущим ingestion. **[tests/test_uncertainty_profiling.py](../../tests/test_uncertainty_profiling.py)** — тесты uncertainty-пути, полезные при изменении его кода. Полный набор не следует объявлять зелёным только потому, что прошли лёгкие примеры guide.

`data/` и `dist/` создаются локальными командами; это не отсутствующие исходные модули, которые требуется найти. Новые файлы самого guide собраны в его [оглавлении](../README.md) и [описании лабораторных](../labs/README.md).
