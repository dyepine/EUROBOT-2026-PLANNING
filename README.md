# EUROBOT-2026-PLANNING

Минимальный runnable POC для проверки стратегии Eurobot 2026:
- симуляция матча на 2D-плоскости;
- utility-based planner для нашего основного робота;
- простой policy-driven соперник;
- опциональные внешние события сценария;
- JSON-результаты и notebook для просмотра траекторий, score и action timeline.

## Структура

```text
poc/
  actions.py
  endgame.py
  entities.py
  external_events.py
  game_state.py
  geometry.py
  main.py
  metrics.py
  opponent_policy.py
  planner.py
  scoring.py
  scenarios.py
  semantic_map.py
  simulator.py
  visualize.py
notebooks/
  poc_results_overview.ipynb
docs/
  codex_poc_strategy_spec.md
```

## Быстрый старт

Требуется `python3` и `matplotlib`.

Запуск одного матча:

```bash
python3 -m poc.main --scenario baseline --output runs/baseline.json
```

Сценарий с отложенным появлением источников:

```bash
python3 -m poc.main --scenario delayed_sources --output runs/delayed_sources.json
```

Небольшой batch:

```bash
python3 -m poc.main --scenario aggressive_enemy --batch 5
```

Доступные сценарии:
- `baseline`
- `delayed_sources`
- `aggressive_enemy`
- `thermo_first_enemy`

## Что уже есть в каркасе

- Семантическая карта поля с источниками, кладовыми, гнёздами и термометром.
- Endgame-конфиг, совместимый с текущим `ChillSequence`.
- Константная модель таймингов по типу действия.
- Fixed-timestep симулятор.
- Сохранение результатов матча в JSON.
- Notebook для интерактивного анализа.

## Notebook

Откройте [notebooks/poc_results_overview.ipynb](/home/napalkov/coding/EUROBOT-2026-PLANNING/notebooks/poc_results_overview.ipynb).

В ноутбуке можно:
- прогнать один матч и посмотреть summary;
- нарисовать траектории и score progression;
- сохранить результат в `runs/`;
- посчитать усреднённые метрики по нескольким seed.

## Дальше по проекту

Следующие логичные шаги:
- уточнить scoring и swing-эвристику;
- добавить больше сценариев и stochastic noise;
- улучшить attack logic и учёт защищённых зон;
- расширить notebook таблицами и сравнением нескольких planner'ов.
