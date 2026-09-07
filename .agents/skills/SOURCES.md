# Происхождение пяти навыков

Проверено 2026-09-07. Это **адаптации**, не неизменные установки целых пакетов.
Сохранены полезные идеи, убраны обязательные циклы, рекурсивные вызовы навыков,
длинные анкеты, рекламные указания и необязательные платные сервисы.
Разработчики исходников не утверждали качество этих адаптаций на AIMO.

## Superpowers — Jesse Vincent, MIT

Ревизия `b36e0829c6d0140e93cfef2ca599b1b07d4a7797`:
[systematic-debugging](https://github.com/obra/superpowers/blob/b36e0829c6d0140e93cfef2ca599b1b07d4a7797/skills/systematic-debugging/SKILL.md),
[dispatching-parallel-agents](https://github.com/obra/superpowers/blob/b36e0829c6d0140e93cfef2ca599b1b07d4a7797/skills/dispatching-parallel-agents/SKILL.md),
[verification-before-completion](https://github.com/obra/superpowers/blob/b36e0829c6d0140e93cfef2ca599b1b07d4a7797/skills/verification-before-completion/SKILL.md).

Перенесено в aimo-implement/review и правила координации: найти причину ошибки,
давать независимым исполнителям узкий контекст, подтверждать результат проверкой.
НЕ перенесены обязательные фазы и требование заново запускать все тесты ради каждого ответа.

## Scientific Agent Skills — K-Dense Inc., MIT

Ревизия `9cf7d9aea7d84754db4c167ab04b299d33c444bc`:
[scientific-brainstorming](https://github.com/K-Dense-AI/scientific-agent-skills/blob/9cf7d9aea7d84754db4c167ab04b299d33c444bc/skills/scientific-brainstorming/SKILL.md),
[scientific-critical-thinking](https://github.com/K-Dense-AI/scientific-agent-skills/blob/9cf7d9aea7d84754db4c167ab04b299d33c444bc/skills/scientific-critical-thinking/SKILL.md),
[literature-review](https://github.com/K-Dense-AI/scientific-agent-skills/blob/9cf7d9aea7d84754db4c167ab04b299d33c444bc/skills/literature-review/SKILL.md).

Перенесено в aimo-research/review/literature: независимая генерация объяснений,
контрпримеры, соразмерная критика, проверка цитат и различение данных/интерпретаций.
НЕ перенесены большой комплект научных инструментов, обязательные схемы и OpenRouter.
Полные лицензионные уведомления обоих проектов: [LICENSES.txt](LICENSES.txt).
`aimo-experiment` — наша небольшая процедура для кода этого репозитория.

## Официальный механизм Codex

[AGENTS](https://learn.chatgpt.com/docs/agent-configuration/agents-md),
[подагенты](https://learn.chatgpt.com/docs/agent-configuration/subagents),
[навыки](https://learn.chatgpt.com/docs/build-skills),
[конфигурация](https://learn.chatgpt.com/docs/config-file/config-reference),
[модели](https://learn.chatgpt.com/docs/models),
[память](https://learn.chatgpt.com/docs/customization/memories).

Актуальные документы подтверждают проектные .agents/skills и .codex/agents/*.toml;
полный SKILL.md читается по выбору, прочие документы сами не попадают в контекст.
Файл роли фиксирует model/effort и имеет приоритет над соответствующими spawn-значениями.
Роли/уровни и ограничение в два помощника — выбор пользователя и этого проекта,
не результат сравнения моделей на AIMO. Конфигурация не отменяет клиентские/аккаунтные ограничения.
Проверены файлы и тесты, но живые переключения моделей/навыков в Codex здесь не измерялись.
