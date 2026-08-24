#!/bin/bash
#SBATCH --job-name=gm_macro
#SBATCH --account=acd109125
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=48:00:00
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

module purge

# batch shell 沒有 conda 這個函式，直接叫 env 裡的 python
PY=${GM_PY:-}
if [ -z "$PY" ]; then
    for c in ~/miniconda3/envs/GraphMaker/bin/python ~/.conda/envs/GraphMaker/bin/python; do
        [ -x "$c" ] && { PY=$c; break; }
    done
fi
[ -n "$PY" ] && [ -x "$PY" ]     || { echo "[ERROR] 找不到 GraphMaker 環境的 python，用 GM_PY 指定"; exit 1; }
echo "python: $PY"

# dgl.sparse 是 C++ 擴充，載入時要 libnvrtc.so.12。torch 用 RPATH 載入自己帶的
# 那份，dgl 的擴充沒有那個設定，所以要把 pip 裝進來的 CUDA 函式庫目錄補上，
# 否則會報 ImportError: Cannot load DGL C++ sparse library。
_NVLIB=$("$PY" -c "import nvidia,os,glob;print(':'.join(glob.glob(os.path.dirname(nvidia.__file__)+'/*/lib')))" 2>/dev/null || true)
[ -n "$_NVLIB" ] && export LD_LIBRARY_PATH="${_NVLIB}:${LD_LIBRARY_PATH:-}"

IDX=${SLURM_ARRAY_TASK_ID:?這支要用 sbatch 送，或加 --list 看組合表}
[ "$IDX" -lt "$N_COMBOS" ] || { echo "[ERROR] index $IDX 超出範圍（共 $N_COMBOS 個）"; exit 1; }

LINE=$(echo "$COMBOS" | sed -n "$((IDX + 1))p")
GROUP=$(echo "$LINE" | awk '{print $1}')
VARIANT=$(echo "$LINE" | awk '{print $2}')
SEED=$(echo "$LINE" | awk '{print $3}')

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export GM_SEED=$SEED

# 生成圖存成與其他模型同結構的序列，才能共用分析工具。
# GraphMaker 生成的是單一靜態圖，每張會被複製 GM_SEQ_LEN 次。
export GM_GEN_DIR="${GM_GEN_DIR:-$HOME/gm_generated}"
export GM_RUN_TAG="${GROUP}_${VARIANT}_seed${SEED}"

mkdir -p logs

echo "=================================================="
echo " array index ${IDX} / ${N_COMBOS}"
echo " dataset : ${GROUP}"
echo " variant : ${VARIANT}"
echo " seed    : ${SEED}"
echo "=================================================="

mkdir -p results
CPT_KEEP="results/${GROUP}_${VARIANT}_seed${SEED}.pth"

# GM_SKIP_TRAIN=1 時，已經存過帶 seed 的 checkpoint 就直接取樣。
# 用在取樣階段失敗、訓練不必重來的情況。
if [ -n "${GM_SKIP_TRAIN:-}" ] && [ -f "$CPT_KEEP" ]; then
    echo
    echo "===== TRAIN 略過（GM_SKIP_TRAIN，沿用 ${CPT_KEEP}）====="
    SKIP_TRAIN=1
else
    echo
    echo "===== TRAIN ====="
    $PY train_${VARIANT}.py -d "$GROUP"
    SKIP_TRAIN=0
fi

# 同一個組合的兩個 variant 會寫進同一個目錄，只依時間取最新會抓錯。
# train_sync.py 存成 Sync_*.pth、train_async.py 存成 Async_*.pth。
case "$VARIANT" in
    sync)  _pat="Sync_*.pth" ;;
    async) _pat="Async_*.pth" ;;
    *)     _pat="*.pth" ;;
esac
if [ "$SKIP_TRAIN" = "1" ]; then
    CPT="$CPT_KEEP"
else
    CPT=$(ls -t ${GROUP}_cpts/${_pat} 2>/dev/null | head -1)
    [ -n "$CPT" ] || { echo "[ERROR] 找不到 checkpoint"; exit 1; }
    # checkpoint 的檔名裡沒有 seed，同組合不同 seed 會互相覆蓋。
    # 另存成帶 seed 的名字，掃描與後續分析才對得起來。
    cp "$CPT" "$CPT_KEEP"
fi
echo "checkpoint: $CPT"

echo
echo "===== SAMPLE ====="
$PY sample.py --model_path "$CPT" 2>&1 | tee "results/${GROUP}_${VARIANT}_seed${SEED}.log"

# set -eo pipefail 之下，sample.py 失敗會讓腳本在上一行就中止，
# 所以這個標記只會在取樣真的跑完時出現。log 裡的關鍵字不能當判準——
# 失敗的 traceback 也會提到 Evaluator。
touch "results/${GROUP}_${VARIANT}_seed${SEED}.done"

echo
echo "===== DONE  ${GROUP} / ${VARIANT} / seed ${SEED} ====="
