# Демонстрационные данные

`sEEG/brain_toy.csv` — читаемое текстовое описание полностью синтетического набора для автономной демонстрации конвейера мозга. Каждая строка задаёт контакт и параметры искусственного события. При запуске конвейер детерминированно разворачивает это компактное описание в необходимые временные ряды в памяти. Файл не содержит измерений пациента и не предназначен для научных выводов или сверки с `reference_results/`.

Пересоздание файла:

```bash
python scripts/generate_brain_toy.py
```

Быстрый демонстрационный запуск:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_brain_toy.py
python src/brain_seeg_pipeline.py \
  --data data/sEEG/brain_toy.csv \
  --out results_quick/brain_toy --seeds 0 --epochs 2
```

Команды выполняются из корня репозитория. На Windows замените активацию окружения на `.venv\Scripts\Activate.ps1`. Результаты записываются в `results_quick/brain_toy/`.

Наборы авиации и финансов не заменяются: их по-прежнему получает `scripts/fetch_data.py` из указанных в основном README источников.
