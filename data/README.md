# Демонстрационные данные / Demonstration data

## Откуда берутся данные мозга / Where the brain data come from

**RU.** Для примера B есть два набора разного происхождения; их нельзя смешивать.

1. **Настоящий замороженный срез** — `sEEG-HFOs-8_cases_derived.npz` (182 МБ, MD5 `6e768cf1cc3a664ac287fc836d19ed62`). Цепочка получения:
   1. Источник: запись sEEG пациента `sEEG-HFOs-8.edf` (Zenodo, DOI 10.5281/zenodo.21967993, CC BY 4.0; 743 МБ, MD5 `c83e85316860502327f6b7ed7ccb15d2`; 103 сигнала, 256 Гц, аннотации врача). В компьютере автора она лежит в репозитории `BioMedAI-sEEG-core-of-epilepsy/dataset/`.
   2. Скрипт [`src/brain_extract_edf.py`](../src/brain_extract_edf.py) (только NumPy) читает EDF и «замораживает» компактный срез: запись обрезается на 10 550 с (дальше — хирургическая каутеризация); для каждого контакта и секунды — лог-мощность в бинах 1–67 Гц (`logspec`), длина линии, ст. откл. после удаления тренда, автокорреляция; сырой сигнал 10 000–10 550 с (`raw_event`); корреляция контактов на фоне 0–3600 с (`corr_base`); аннотации врача. В файл записывается MD5 исходного EDF.
   3. Конвейер [`src/brain_seeg_pipeline.py`](../src/brain_seeg_pipeline.py) читает только этот срез и проверяет записанный в нём MD5 EDF.
   4. Пути задаёт `config/paths.json` (`seeg_edf`, `seeg_derived`) или `.env` (`DISSER_SEEG_EDF`, `DISSER_SEEG_DERIVED`); `python scripts/fetch_data.py` ищет срез и, если его нет, строит из EDF (`--download-edf` скачивает EDF).
   Срез и EDF не хранятся в этом репозитории; замороженные результаты расчёта по ним лежат в `reference_results/brain/`, и именно они использованы в рисунках примера B.

2. **`sEEG/brain_toy.csv` — синтетический двойник, не срез реальных данных.** Файл (20 контактов) создан скриптом [`scripts/generate_brain_toy.py`](../scripts/generate_brain_toy.py): параметры контактов и события (время начала, частота, амплитуда) заданы в самом скрипте, а все ряды генерируются из случайных чисел с фиксированной затравкой `20260924`. Из записи пациента в него не скопировано ни одного значения. От реального среза повторены только **форма и временная шкала** — имена контактов в формате «стержень + номер» (`R1`, `FD1`, `PA'5`, …), частота 256 Гц, обрезка 10 550 с, фон 0–3600 с, окно сырого сигнала 10 000–10 550 с, отметка события 10 396,445 с, конец приступа 10 452 с, — чтобы конвейер мозга выполнялся без загрузки 743 МБ. Поэтому выводы и сверка с `reference_results/` по нему невозможны.

**EN.** Example B uses two datasets of different origin; they must not be confused.

1. **The real frozen slice** — `sEEG-HFOs-8_cases_derived.npz` (182 MB, MD5 `6e768cf1cc3a664ac287fc836d19ed62`). Chain of custody:
   1. Source: a patient's sEEG recording `sEEG-HFOs-8.edf` (Zenodo, DOI 10.5281/zenodo.21967993, CC BY 4.0; 743 MB, MD5 `c83e85316860502327f6b7ed7ccb15d2`; 103 signals, 256 Hz, clinician annotations). On the author's machine it sits in the `BioMedAI-sEEG-core-of-epilepsy/dataset/` repository.
   2. [`src/brain_extract_edf.py`](../src/brain_extract_edf.py) (NumPy only) reads the EDF and freezes a compact slice: the recording is cropped at 10 550 s (surgical cautery follows); per contact and second — log-power in 1–67 Hz bins (`logspec`), line length, detrended std, lag-1 autocorrelation; the raw signal 10 000–10 550 s (`raw_event`); the contact correlation on the 0–3600 s baseline (`corr_base`); the clinician annotations. The MD5 of the source EDF is stored inside the file.
   3. [`src/brain_seeg_pipeline.py`](../src/brain_seeg_pipeline.py) reads only this slice and checks the stored EDF MD5.
   4. Paths come from `config/paths.json` (`seeg_edf`, `seeg_derived`) or `.env` (`DISSER_SEEG_EDF`, `DISSER_SEEG_DERIVED`); `python scripts/fetch_data.py` looks for the slice and builds it from the EDF if absent (`--download-edf` fetches the EDF).
   Neither the slice nor the EDF is stored in this repository; the frozen results computed from them are in `reference_results/brain/`, and the example-B figures use exactly those.

2. **`sEEG/brain_toy.csv` is a synthetic stand-in, not a slice of real data.** The 20-contact file is created by [`scripts/generate_brain_toy.py`](../scripts/generate_brain_toy.py): contact and event parameters (onset time, frequency, amplitude) are written in the script itself, and all series are generated from random numbers with the fixed seed `20260924`. Not one value is copied from the patient's recording. Only the **shape and timeline** of the real slice are mimicked — contact names in "shaft + number" form (`R1`, `FD1`, `PA'5`, …), 256 Hz, the 10 550 s crop, the 0–3600 s baseline, the 10 000–10 550 s raw window, the event mark at 10 396.445 s, the seizure end at 10 452 s — so that the brain pipeline runs without the 743 MB download. No conclusions or comparison with `reference_results/` can be drawn from it.

---

Ниже — `sEEG/brain_toy.csv`: читаемое текстовое описание полностью синтетического набора для автономной демонстрации конвейера мозга. Каждая строка задаёт контакт и параметры искусственного события. При запуске конвейер детерминированно разворачивает это компактное описание в необходимые временные ряды в памяти. Файл не содержит измерений пациента и не предназначен для научных выводов или сверки с `reference_results/`.

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
