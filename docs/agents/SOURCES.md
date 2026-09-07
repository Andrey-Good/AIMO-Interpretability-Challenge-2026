# Источники и границы утверждений

Проверено 7 сентября 2026 года. Ниже первичные источники OpenAI; это не поиск по
старым статьям о настройке Claude Code или неофициальным «советам по Codex».
Старые адреса `developers.openai.com/codex/...` сейчас перенаправляют на ChatGPT Learn.
Версия клиента и доступ аккаунта могут отставать от документации. Конфигурация
проверена синтаксически, но её загрузка реальным клиентом в этой среде не проверялась.

## Настройка и работа Codex

- [AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md): область
  действия инструкций, приоритет близких файлов, ограничение объёма. Наш AGENTS —
  короткая карта обязательных правил; подробные процедуры загружаются по необходимости.
- [Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents): отдельные
  `.codex/agents/*.toml`, обязательные name/description/developer_instructions,
  наследование модели и reasoning. Поэтому роли фиксируют оба параметра явно.
- [Build skills](https://learn.chatgpt.com/docs/build-skills): проектная папка
  `.agents/skills`, YAML-заголовок SKILL.md, постепенная загрузка, явный вызов,
  `allow_implicit_invocation`. Навык — инструкция, не новая модель и не доступ к GPU.
- [Configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference):
  только доверенные проектные конфиги; актуальные ключи agents, sandbox и approvals.
  В проектном config игнорируются profile/profiles, model_provider, notify и другие
  машинные настройки. Мы не пытаемся менять провайдера или ключи из проекта.
- [Advanced configuration](https://learn.chatgpt.com/docs/config-file/config-advanced):
  профиль теперь отдельный `~/.codex/name.config.toml`, выбираемый `--profile name`.
  Пример со старой моделью на странице не использован как рекомендация модели.
- [Models](https://learn.chatgpt.com/docs/models): актуальные имена моделей Codex,
  выбор через `/model` или `codex -m`, уровни размышления и режим Ultra. Последний
  не следует путать с отдельным API-значением reasoning effort.
- [Memories](https://learn.chatgpt.com/docs/customization/memories): личная память
  Codex в `~/.codex/memories/`. `MEMORY/` здесь — наша версия командной памяти,
  которую агент читает потому, что это предписано AGENTS, а не из-за магического имени.
- [Long-running work](https://learn.chatgpt.com/docs/long-running-work): ограниченная
  проверяемая цель, `/goal`, наблюдаемое завершение. Не обещание бесконечного сервера.
- [Git worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees): отдельные
  рабочие папки для параллельных задач; это не изоляция GPU или сети.

## Astra / Terra / Luna / Sol

- [Using GPT-6 Astra](https://developers.openai.com/api/docs/guides/latest-model):
  ясные инструкции, осмысленное делегирование, проверка изменённого поведения,
  ограничение лишних уточнений для обратимых задач. Страница latest-model меняется;
  при проверке отдельный её HTML-ответ показывал старый раздел, поэтому имена и
  уровни дополнительно сверены с каталогом Codex и точными страницами моделей.
- [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).
- [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra).
- [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna).
- [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol).

Цены API не означают стоимость лимитов подписки Codex. Роли и уровни в этом проекте —
начальная инженерная политика, **не опубликованный OpenAI результат сравнения этих
моделей на AIMO**. «Luna всегда справляется» и «Sol стоит как Astra» не установлено.
Проверять по своим задачам: принятый результат, переделки, задержка и фактический расход.

## Наука и проектирование среды

- [Harness engineering](https://openai.com/index/harness-engineering/): компактный
  AGENTS как оглавление, версия знаний внутри репозитория и проверки вместо огромного
  постоянно загружаемого руководства. Наши конкретные лимиты/шаблоны разработаны здесь.
- [ChatGPT for academic researchers](https://openai.com/index/chatgpt-for-academic-researchers/):
  обсуждение гипотез/литературы и проверяемые вычислительные задачи с Codex;
  научное суждение и ответственность исследователя остаются необходимыми.
- [Accelerating science with GPT-5](https://openai.com/index/accelerating-science-gpt-5/):
  исторические примеры 2025 года для понимания рабочего цикла, **не руководство по
  текущим моделям и конфигам**. Не используется как источник версии Codex.
- [Official skills](https://github.com/openai/skills/tree/main/skills/.curated): просмотрен
  каталог; установлен define-goal с происхождением и лицензией. Сторонние исполняемые
  пакеты не устанавливались. Узкие навыки AIMO написаны специально для этого проекта.

## Что является нашим выбором, а не официальным стандартом

Два помощника одновременно; Astra как научный координатор; запрет рекурсивного
делегирования; один тяжёлый опыт на репозиторий; подготовительный и итоговый коммиты;
отдельные журналы отрицательных опытов; группировка данных по исходной задаче;
статусы памяти; заданные ниже контракты тензоров. Это обоснованные проектные правила,
которые надо корректировать по результатам, а не выдавать за требования OpenAI.

Не гарантируется отсутствие ошибок. Тесты проверяют конкретные свойства;
согласие двух агентов не является доказательством. Текущие правила/датасеты AIMO
перед конкурсным запуском отдельно сверяются с организаторами.
