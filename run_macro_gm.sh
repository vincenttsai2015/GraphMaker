#!/bin/bash
#SBATCH --job-name=gm_macro
#SBATCH --account=ACD109125
#SBATCH --partition=gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# GraphMaker 跑巨觀動態資料。訓練完接著取樣。
#
#   sbatch --array=0-23 run_macro_gm.sh       前 24 個（每個組合各一個 seed）
#   bash run_macro_gm.sh --list               印出組合對照表
#
# 組合清單一行一個：<資料集組合> <variant> <seed>，順序是 seed 最外層。
#
# GraphMaker 的 set_seed() 寫死 0，這支用 GM_SEED 傳進去，
# train_sync.py / sample.py 會讀它。

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

COMBOS='macro_superuser_burst_a2q sync 0
macro_superuser_burst_a2q async 0
macro_superuser_burst_hysteresis_a2q sync 0
macro_superuser_burst_hysteresis_a2q async 0
macro_superuser_hysteresis_a2q sync 0
macro_superuser_hysteresis_a2q async 0
macro_superuser_raw_a2q sync 0
macro_superuser_raw_a2q async 0
macro_twitter_burst_RT sync 0
macro_twitter_burst_RT async 0
macro_twitter_burst_hysteresis_RT sync 0
macro_twitter_burst_hysteresis_RT async 0
macro_twitter_hysteresis_RT sync 0
macro_twitter_hysteresis_RT async 0
macro_twitter_raw_RT sync 0
macro_twitter_raw_RT async 0
macro_wiki_vote_burst_hysteresis_support sync 0
macro_wiki_vote_burst_hysteresis_support async 0
macro_wiki_vote_burst_support sync 0
macro_wiki_vote_burst_support async 0
macro_wiki_vote_hysteresis_support sync 0
macro_wiki_vote_hysteresis_support async 0
macro_wiki_vote_raw_support sync 0
macro_wiki_vote_raw_support async 0
macro_superuser_burst_a2q sync 1
macro_superuser_burst_a2q async 1
macro_superuser_burst_hysteresis_a2q sync 1
macro_superuser_burst_hysteresis_a2q async 1
macro_superuser_hysteresis_a2q sync 1
macro_superuser_hysteresis_a2q async 1
macro_superuser_raw_a2q sync 1
macro_superuser_raw_a2q async 1
macro_twitter_burst_RT sync 1
macro_twitter_burst_RT async 1
macro_twitter_burst_hysteresis_RT sync 1
macro_twitter_burst_hysteresis_RT async 1
macro_twitter_hysteresis_RT sync 1
macro_twitter_hysteresis_RT async 1
macro_twitter_raw_RT sync 1
macro_twitter_raw_RT async 1
macro_wiki_vote_burst_hysteresis_support sync 1
macro_wiki_vote_burst_hysteresis_support async 1
macro_wiki_vote_burst_support sync 1
macro_wiki_vote_burst_support async 1
macro_wiki_vote_hysteresis_support sync 1
macro_wiki_vote_hysteresis_support async 1
macro_wiki_vote_raw_support sync 1
macro_wiki_vote_raw_support async 1
macro_superuser_burst_a2q sync 2
macro_superuser_burst_a2q async 2
macro_superuser_burst_hysteresis_a2q sync 2
macro_superuser_burst_hysteresis_a2q async 2
macro_superuser_hysteresis_a2q sync 2
macro_superuser_hysteresis_a2q async 2
macro_superuser_raw_a2q sync 2
macro_superuser_raw_a2q async 2
macro_twitter_burst_RT sync 2
macro_twitter_burst_RT async 2
macro_twitter_burst_hysteresis_RT sync 2
macro_twitter_burst_hysteresis_RT async 2
macro_twitter_hysteresis_RT sync 2
macro_twitter_hysteresis_RT async 2
macro_twitter_raw_RT sync 2
macro_twitter_raw_RT async 2
macro_wiki_vote_burst_hysteresis_support sync 2
macro_wiki_vote_burst_hysteresis_support async 2
macro_wiki_vote_burst_support sync 2
macro_wiki_vote_burst_support async 2
macro_wiki_vote_hysteresis_support sync 2
macro_wiki_vote_hysteresis_support async 2
macro_wiki_vote_raw_support sync 2
macro_wiki_vote_raw_support async 2
macro_superuser_burst_a2q sync 3
macro_superuser_burst_a2q async 3
macro_superuser_burst_hysteresis_a2q sync 3
macro_superuser_burst_hysteresis_a2q async 3
macro_superuser_hysteresis_a2q sync 3
macro_superuser_hysteresis_a2q async 3
macro_superuser_raw_a2q sync 3
macro_superuser_raw_a2q async 3
macro_twitter_burst_RT sync 3
macro_twitter_burst_RT async 3
macro_twitter_burst_hysteresis_RT sync 3
macro_twitter_burst_hysteresis_RT async 3
macro_twitter_hysteresis_RT sync 3
macro_twitter_hysteresis_RT async 3
macro_twitter_raw_RT sync 3
macro_twitter_raw_RT async 3
macro_wiki_vote_burst_hysteresis_support sync 3
macro_wiki_vote_burst_hysteresis_support async 3
macro_wiki_vote_burst_support sync 3
macro_wiki_vote_burst_support async 3
macro_wiki_vote_hysteresis_support sync 3
macro_wiki_vote_hysteresis_support async 3
macro_wiki_vote_raw_support sync 3
macro_wiki_vote_raw_support async 3
macro_superuser_burst_a2q sync 4
macro_superuser_burst_a2q async 4
macro_superuser_burst_hysteresis_a2q sync 4
macro_superuser_burst_hysteresis_a2q async 4
macro_superuser_hysteresis_a2q sync 4
macro_superuser_hysteresis_a2q async 4
macro_superuser_raw_a2q sync 4
macro_superuser_raw_a2q async 4
macro_twitter_burst_RT sync 4
macro_twitter_burst_RT async 4
macro_twitter_burst_hysteresis_RT sync 4
macro_twitter_burst_hysteresis_RT async 4
macro_twitter_hysteresis_RT sync 4
macro_twitter_hysteresis_RT async 4
macro_twitter_raw_RT sync 4
macro_twitter_raw_RT async 4
macro_wiki_vote_burst_hysteresis_support sync 4
macro_wiki_vote_burst_hysteresis_support async 4
macro_wiki_vote_burst_support sync 4
macro_wiki_vote_burst_support async 4
macro_wiki_vote_hysteresis_support sync 4
macro_wiki_vote_hysteresis_support async 4
macro_wiki_vote_raw_support sync 4
macro_wiki_vote_raw_support async 4'
N_COMBOS=120

if [ "$1" = "--list" ]; then
    echo "共 ${N_COMBOS} 個組合"
    printf "%5s  %-44s %-7s %s\n" idx group variant seed
    echo "$COMBOS" | awk '{ printf "%5d  %-44s %-7s %s\n", NR-1, $1, $2, $3 }'
    exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID:?這支要用 sbatch 送，或加 --list 看組合表}
[ "$IDX" -lt "$N_COMBOS" ] || { echo "[ERROR] index $IDX 超出範圍（共 $N_COMBOS 個）"; exit 1; }

LINE=$(echo "$COMBOS" | sed -n "$((IDX + 1))p")
GROUP=$(echo "$LINE" | awk '{print $1}')
VARIANT=$(echo "$LINE" | awk '{print $2}')
SEED=$(echo "$LINE" | awk '{print $3}')

module load miniconda3/conda24.5.0_py3.9
PY=~/.conda/envs/GraphMaker/bin/python
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export GM_SEED=$SEED

mkdir -p logs

echo "=================================================="
echo " array index ${IDX} / ${N_COMBOS}"
echo " dataset : ${GROUP}"
echo " variant : ${VARIANT}"
echo " seed    : ${SEED}"
echo "=================================================="

echo
echo "===== TRAIN ====="
$PY train_${VARIANT}.py -d "$GROUP"

CPT=$(ls -t ${GROUP}_cpts/*.pth 2>/dev/null | head -1)
[ -n "$CPT" ] || { echo "[ERROR] 找不到 checkpoint"; exit 1; }
echo "checkpoint: $CPT"

echo
echo "===== SAMPLE ====="
$PY sample.py --model_path "$CPT"
