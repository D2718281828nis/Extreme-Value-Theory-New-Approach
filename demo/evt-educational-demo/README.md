# EVT Educational Demo

Автономный учебный модуль показывает Block Maxima/GEV, Peaks Over Threshold/GPD и сравнение точечной POT-детекции с простой AR-моделью остатков. Все данные синтетические; результаты не следует переносить на прикладные риски без проверки предпосылок.

## Быстрый запуск

Из корня основного репозитория:

```bash
python3 -m venv demo/evt-educational-demo/.venv
source demo/evt-educational-demo/.venv/bin/activate
python -m pip install -e 'demo/evt-educational-demo[dev]'
python demo/evt-educational-demo/scripts/run_demo.py --seed 42
```

На Windows используйте `.venv\Scripts\Activate.ps1`. Команда создаёт `data/synthetic_data.csv`, печатает метрики и возвратные уровни, а также сохраняет семь PNG в `figures/`.

## Проверка

```bash
cd demo/evt-educational-demo
python -m pytest -q
jupyter nbconvert --execute --to notebook --inplace notebooks/01_introduction_to_evt.ipynb
```

Остальные ноутбуки запускаются аналогично. После проверки outputs следует очистить перед коммитом.

## Методы

| Метод | Задача | Ограничение |
|---|---|---|
| GEV Block Maxima | Максимумы блоков, возвратные уровни | Теряет неблочные экстремумы; период задаётся в блоках |
| GPD POT | Превышения высокого порога, точечная детекция | Результат чувствителен к порогу и зависимости превышений |
| AR residual baseline | Динамика среднего и аномальные остатки | Порог $k\sigma$ — эвристика, а не хвостовая EVT-модель |

Для GEV используются положение $\mu$, масштаб $\sigma>0$ и форма $\xi$. В `scipy.stats.genextreme` параметр `c=-xi`; код явно преобразует знак. Для GPD применяется `scipy.stats.genpareto` с фиксированным `loc=0`.

## Структура

- `src/evt_demo/` — библиотечный код;
- `scripts/run_demo.py` — полный воспроизводимый сценарий;
- `tests/` — unit- и smoke-тесты;
- `notebooks/` — четыре учебных занятия;
- `data/` и `figures/` — воспроизводимые артефакты.

## Воспроизводимость и ограничения

Генератор использует `numpy.random.default_rng(seed)`. Фиксированный seed воспроизводит CSV; bootstrap также принимает seed. Синтетические импульсы специально упрощены, превышения могут быть зависимы, а доверительные интервалы являются учебной параметрической оценкой.

## Научная основа

Fisher & Tippett (1928), Gnedenko (1943), Coles (2001), de Haan & Ferreira (2006); API: [SciPy `genextreme`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html) и [SciPy `genpareto`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html).
