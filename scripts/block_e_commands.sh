#!/usr/bin/env bash
# Повторный прогон расчётов блока E § 3.11 (обучение ГНС на графе объекта) кодом репозитория автора.
#
# Запускать ИЗ КОРНЯ репозитория BioMedAI-sEEG-core-of-epilepsy в его собственном окружении:
#   cd ../BioMedAI-sEEG-core-of-epilepsy
#   python3 -m venv .venv && source .venv/bin/activate && pip install -e .
#   bash ../dissertation-calculations/scripts/block_e_commands.sh
# Затем сверка:
#   cd ../dissertation-calculations && python scripts/check_block_e.py --results gnn_model_result_rerun
#
# Все результаты пишутся в *_rerun/, сохранённые автором gnn_model_result/ и object_model_result/ не перезаписываются.
# Команды и параметры взяты из README репозитория (раздел gnn_model) и из сохранённых JSON (model_config).
# Прогоны с группировкой по стволу чувствительны к затравке и к BLAS (OpenBLAS / Apple Accelerate) —
# на macOS результат затравки 7 может отличаться от сохранённого (см. README репозитория и блок E).
set -euo pipefail
G=object_model_result/sEEG-HFOs-8/object_model_graph.graphml
GD=object_model_result_rerun/sEEG-HFOs-8/object_model_graph_dfa.graphml
O=gnn_model_result_rerun
DEEP="--architecture gat --heads 2 --num-layers 3 --hidden-channels 8 --residual --dropout 0.4 --drop-edge-p 0.2 \
      --weight-decay 0.2 --early-stopping-patience 60 --early-stopping-metric val_macro_f1 --epochs 300"

# 1. Четыре конфигурации на одиночном разбиении 67/30
python -m gnn_model.run_gnn --graph $G --output $O/baseline_overfit --epochs 150
python -m gnn_model.run_gnn --graph $G --output $O/regularized --hidden-channels 8 --dropout 0.6 \
  --weight-decay 1e-2 --drop-edge-p 0.3 --early-stopping-patience 20 --epochs 300
python -m gnn_model.run_gnn --graph $G --output $O/attention --architecture gat --heads 1 --num-layers 1 \
  --dropout 0.5 --drop-edge-p 0.2 --early-stopping-patience 40 --early-stopping-metric val_macro_f1 --epochs 300
python -m gnn_model.run_gnn --graph $G --output $O/attention_deep $DEEP

# 2. Признак DFA по 30-секундной базовой линии (нужен исходный EDF)
mkdir -p "$(dirname $GD)"
python -m gnn_model.augment_dfa --edf dataset/sEEG-HFOs-8.edf --graph $G --output $GD --baseline-seconds 30

# 3. Пятикратная кросс-валидация: обычная и с группировкой по электродному стволу, без DFA и с DFA
python -m gnn_model.run_gnn --graph $G  --output $O/attention_deep_cv           $DEEP --cross-validate 5
python -m gnn_model.run_gnn --graph $GD --output $O/attention_deep_dfa_cv       $DEEP --cross-validate 5
python -m gnn_model.run_gnn --graph $G  --output $O/attention_deep_cv_shaft     $DEEP --cross-validate 5 --group-by-shaft
python -m gnn_model.run_gnn --graph $GD --output $O/attention_deep_dfa_cv_shaft $DEEP --cross-validate 5 --group-by-shaft

# 4. Пять затравок для группировки по стволу (таблица «полнота по ранним / ложные срабатывания» блока E)
for s in 1 2 3 7 11; do
  python -m gnn_model.run_gnn --graph $G  --output $O/seeds/shaft_seed$s     $DEEP --cross-validate 5 --group-by-shaft --seed $s
  python -m gnn_model.run_gnn --graph $GD --output $O/seeds/dfa_shaft_seed$s $DEEP --cross-validate 5 --group-by-shaft --seed $s
done
echo "Готово. Матрицы вне фолда по затравкам: $O/seeds/*/sEEG-HFOs-8/gnn_cv_result.json (out_of_fold_confusion_matrix)"
