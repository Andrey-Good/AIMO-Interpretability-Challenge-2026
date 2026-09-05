# Библиотеки через то, что ты уже знаешь

[Начало](../README.md). Здесь не требуется предварительно проходить целый курс. Сначала пойми роль конкретного вызова; подробное API понадобится при его изменении.

## Transformers: готовый большой nn.Module

`AutoTokenizer.from_pretrained(...)` загружает токенизатор; `tokenizer(text, return_tensors='pt')` даёт тензоры ID и маску. `AutoModelForCausalLM.from_pretrained(...)` создаёт PyTorch-модель с головой предсказания следующего токена и загружает веса. `model(**inputs)` — знакомый forward, `model.generate(**inputs)` — цикл autoregressive-продолжения. Это разные операции: probe использует первую, uncertainty — вторую.

`output_hidden_states=True` добавляет в результат tuple промежуточных представлений. `return_dict=True` позволяет читать именованные поля результата. `apply_chat_template` добавляет служебное оформление ролей; это часть реального входа модели, а не косметика для человека. [Выходы моделей](https://huggingface.co/docs/transformers/main_classes/output), [генерация](https://huggingface.co/docs/transformers/main_classes/text_generation).

`LogitsProcessor` — вызываемый объект внутри генерации, получающий `input_ids` и `scores`. Обычно он может менять scores. Наш `GenerationConfidenceCollector` только измеряет их и возвращает без изменений. Его устройство разобрано [в основной главе](../route/03-uncertainty.md). [Описание processors](https://huggingface.co/docs/transformers/internal/generation_utils).

## Accelerate, Hub, tokenizers и safetensors

`accelerate` помогает Transformers размещать модель при `device_map='auto'` в uncertainty-path; это не отдельная новая архитектура и не автоматическое обучение probe. Загрузка probe устроена проще: `model.to(device)`.

`huggingface-hub` — доступ к файлам моделей и кешу. `local_files_only=True` запрещает загрузчику искать недостающие файлы в сети. Сам кеш не превращает произвольный alias в имя checkpoint: явное соответствие находится в `resolve_checkpoint_model_id`.

`tokenizers` обеспечивает низкоуровневую токенизацию; `safetensors` — формат файлов тензоров весов. Для первого чтения не нужно разбирать их внутренности. Список и версии зависимостей бери из [pyproject.toml](../../pyproject.toml) и [Dockerfile](../../Dockerfile.competition), а не из произвольной свежей инструкции в интернете.

## pandas и Parquet: признаки с именами столбцов

`DataFrame` здесь удобен как матрица признаков вместе с названиями столбцов. `frame[list(FEATURE_NAMES)]` выбирает их **в нужном порядке**. `from_records` собирает строки из словарей. `.iloc[indices]` выбирает строки по позициям, `.groupby(...)` объединяет одинаковые задачи, `.merge(...)` возвращает признаки к повторным исходным строкам. `read_parquet`/`to_parquet` читают и сохраняют табличный кеш; `pyarrow` нужен для этого локального пути, а не для получения `list[bool]` на evaluator.

## scikit-learn: обучаемый маленький предсказатель

`.fit(X,y)` подбирает параметры по признакам и целям; `.predict(X)` применяет уже обученный estimator. Это не обязательно нейросеть. В uncertainty используются ансамбли деревьев.

`Pipeline([imputer, regressor])` объединяет заполнение пропусков и предсказатель. `SimpleImputer(strategy='median')` учит медианы на тренировочной части, затем применяет их к новым строкам. Поэтому pipeline клонируется и обучается заново внутри каждого фолда; нельзя сначала посчитать медианы по всей выборке, включая проверочную часть. [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

`StratifiedGroupKFold` не разделяет одну группу между train и validation и старается сохранять распределение классов. В этом коде группа — текст `original_problem`. [Документация разбиения](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html). Полезно понимать это раньше, чем разбирать все параметры Random Forest.

## Остальное — обычный Python

`dataclass` — объект с объявленными полями, `frozen=True` запрещает обычное переприсваивание полей, но не превращает вложенный словарь в неизменяемый. `Protocol` описывает ожидаемые методы объекта. `joblib`/`pickle` сохраняют Python-объекты; это не обучение и не безопасный формат для неизвестных файлов. `Path(__file__)` привязывает пути к расположению исходника. `importlib` загружает `solution.py` по пути. `argparse` читает аргументы CLI, а `subprocess` запускает отдельный процесс.

Ни один из этих терминов не требует сначала изучить целую библиотеку: возвращайся к API тогда, когда меняешь использующий его участок.
