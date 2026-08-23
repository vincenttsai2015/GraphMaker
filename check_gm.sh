#!/bin/bash
# 掃出哪些組合還沒跑完，並送出。
#
#   bash check_gm.sh                  列出缺漏
#   bash check_gm.sh --submit         送出缺的
#   SEEDS="1" bash check_gm.sh        只看 seed 1（多人分工用）
#   LIMIT=10 bash check_gm.sh --submit
#
# 完成的判準是 results/<組合>_<variant>_seed<S>.done 這個標記檔。
# checkpoint 的檔名沒有 seed，同組合不同 seed 會互相覆蓋，所以不能靠它判斷。

set -eo pipefail
cd "$(dirname "$0")"

RUNNER=${RUNNER:-run_macro_gm_nano4.sh}
LIMIT=${LIMIT:-20}
ACCOUNT=${ACCOUNT:-acd109125}
SUBMIT=0
[ "$1" = "--submit" ] && SUBMIT=1

[ -f "$RUNNER" ] || { echo "[ERROR] 找不到 $RUNNER"; exit 1; }

IN_QUEUE=$( { squeue -u "$USER" -h 2>/dev/null || true; } | wc -l )

# 正在跑的 array index。JOBID 是 <jobid>_<index>，排隊中的會寫成
# <jobid>_[20-23] 或 <jobid>_[20,22]，兩種都要展開。
RUNNING=""
if [ "$IN_QUEUE" -gt 0 ]; then
    for id in $(squeue -u "$USER" -h -o "%i" 2>/dev/null); do
        case "$id" in
            *_\[*\]*)
                spec=${id#*_[}; spec=${spec%]*}; spec=${spec%\%*}
                for part in $(echo "$spec" | tr ',' ' '); do
                    case "$part" in
                        *-*) for k in $(seq "${part%%-*}" "${part##*-}"); do
                                 RUNNING="${RUNNING} ${k}"; done ;;
                        *)   RUNNING="${RUNNING} ${part}" ;;
                    esac
                done ;;
            *_*) RUNNING="${RUNNING} ${id##*_}" ;;
        esac
    done
    echo "queue 裡有 ${IN_QUEUE} 個，對應 $(echo $RUNNING | wc -w) 個 index。"
fi

N_OK=0; N_MISS=0; N_RUN=0
TODO=""

# 組合表由執行腳本自己列出，index 與它一致
LIST=$(bash "$RUNNER" --list 2>/dev/null | tail -n +3)
[ -n "$LIST" ] || { echo "[ERROR] $RUNNER --list 沒有輸出"; exit 1; }

while read -r idx group variant seed; do
    [ -n "$idx" ] || continue
    if [ -n "$SEEDS" ]; then
        case " $SEEDS " in
            *" $seed "*) ;;
            *) continue ;;
        esac
    fi

    # 判準是取樣跑完才寫的標記檔。不能用 log 裡的關鍵字——
    # 失敗的 traceback 也會提到 Evaluator。
    if [ -f "results/${group}_${variant}_seed${seed}.done" ]; then
        N_OK=$((N_OK + 1))
    elif case " $RUNNING " in *" $idx "*) true ;; *) false ;; esac; then
        N_RUN=$((N_RUN + 1))
        printf '%-46s %-6s seed %s   index %-4s 執行中
' "$group" "$variant" "$seed" "$idx"
    else
        N_MISS=$((N_MISS + 1))
        printf '%-46s %-6s seed %s   index %s\n' "$group" "$variant" "$seed" "$idx"
        TODO="${TODO}${idx},"
    fi
done <<EOF
$LIST
EOF

echo
echo "完成 ${N_OK} 組，執行中 ${N_RUN} 組，有缺 ${N_MISS} 組"
[ "$N_MISS" -eq 0 ] && exit 0

TODO=${TODO%,}

# SLURM 的 MaxSubmitJobs 算的是送出數，array 的 %N 只限制同時執行數，
# 24 個 task 一次送出仍會被 QOSMaxSubmitJobPerUserLimit 擋下。
# 所以一次只送剩餘名額那麼多個，其餘等 queue 空出來再執行同一條。
SLOTS=$((LIMIT - IN_QUEUE))
CHUNK=$(echo "$TODO" | cut -d, -f1-"$SLOTS")

echo
echo "queue 現有 ${IN_QUEUE} 個，上限 ${LIMIT}，這次可送 ${SLOTS} 個（待補 ${N_MISS} 個）"
if [ "$SLOTS" -le 0 ]; then
    echo "沒有名額，等 queue 空出來再執行一次"
    exit 0
fi

echo
echo "--- 送出指令 ---"
echo "sbatch --account=${ACCOUNT} --array=${CHUNK} ${RUNNER}"

if [ "$SUBMIT" = "1" ]; then
    echo
    sbatch --account="${ACCOUNT}" --export=ALL,GM_SKIP_TRAIN="${GM_SKIP_TRAIN:-}" --array="${CHUNK}" "${RUNNER}"
    if [ "$N_MISS" -gt "$SLOTS" ]; then
        echo
        echo "還有 $((N_MISS - SLOTS)) 個沒送，等 queue 空出來再執行一次"
    fi
fi
