"""比對生成序列與訓練資料的每張邊數走勢。

    python gm_check.py                    掃全部組合，一組一行
    python gm_check.py macro_wiki_vote_burst_support   看單一組合的完整走勢

判斷有沒有問題看兩個數字。相鄰完全相同的比例要低——100% 表示條件沒進去、
退回同一張複製多次，平台期造成的少量重複是正常的。與真實走勢的相關要是正的，負的或接近 0 表示沒學到時刻條件。
raw 沒注入動態、真實走勢是平的，相關在那裡沒有意義，改看生成的邊數水準
是不是落在真實的兩倍以內。
"""
import glob
import os
import pickle
import sys

import numpy as np

from macro_data import MacroCondDataset

GEN = os.path.expanduser("~/gm_generated")


def real_trace(group):
    d = MacroCondDataset(group)
    T = d.t_target
    n = len(d) // T
    return d.t_split, T, [np.mean([len(d.train[s * T + k]["src"])
                                   for s in range(n)]) for k in range(T)]


def load_seqs(path, T):
    """回傳 (序列 list, 說明)。認不得的格式回傳 (None, 原因)。"""
    g = pickle.load(open(path, "rb"))
    if not isinstance(g, list) or not g:
        return None, f"頂層是 {type(g).__name__}，不是非空 list"
    if hasattr(g[0], "number_of_edges"):
        return None, f"攤平的 {len(g)} 張圖，不是序列（舊版執行留下的）"
    if not isinstance(g[0], list):
        return None, f"元素是 {type(g[0]).__name__}"
    lens = {len(s) for s in g}
    if lens != {T}:
        return None, f"每條張數 {sorted(lens)}，預期 {T}"
    return g, ""


# 真實走勢的變異係數低於這個值就當成平的。raw 沒注入動態，實測 0.02~0.03，
# burst 是 0.33~0.42，中間沒有模糊地帶。
FLAT_CV = 0.10


def stats(seqs, real):
    e = np.array([[x.number_of_edges() for x in s] for s in seqs])
    gen = e.mean(axis=0)
    same = np.mean([set(s[i].edges()) == set(s[i + 1].edges())
                    for s in seqs for i in range(len(s) - 1)])
    real = np.asarray(real, dtype=float)
    flat = real.std() / real.mean() < FLAT_CV if real.mean() else True
    corr = (np.corrcoef(real, gen)[0, 1]
            if np.std(gen) > 0 and not flat else float("nan"))
    # 走勢平的時候相關沒有意義，改看邊數的水準差多少倍。
    ratio = gen.mean() / real.mean() if real.mean() else float("nan")
    return gen, 100 * same, corr, flat, ratio


def one(group):
    t_split, T, real = real_trace(group)
    pats = sorted(glob.glob(f"{GEN}/{group}_*/GraphMaker/sampled_ts.pkl"))
    if not pats:
        sys.exit(f"找不到 {group} 的產出")

    print(f"{'':18s}" + " ".join(f"{t_split + k:4d}" for k in range(T)))
    print(f"{'真實':<16s}" + " ".join(f"{v:4.0f}" for v in real))
    print()
    for p in pats:
        tag = p.rsplit(os.sep, 3)[1]
        print(f"--- {tag} ---")
        seqs, why = load_seqs(p, T)
        if seqs is None:
            print(f"    [ERROR] {why}")
            continue
        gen, same, corr, flat, ratio = stats(seqs, real)
        print(f"{'生成':<16s}" + " ".join(f"{v:4.0f}" for v in gen))
        print(f"    {len(seqs)} 條 x {len(seqs[0])} 張，"
              f"節點 {seqs[0][0].number_of_nodes()}")
        tail = (f"邊數水準 {ratio:.2f}x（真實走勢是平的，相關無意義）"
                if flat else f"與真實走勢相關 {corr:+.3f}")
        print(f"    相鄰完全相同 {same:.1f}%   {tail}")


def sweep():
    pats = sorted(glob.glob(f"{GEN}/*/GraphMaker/sampled_ts.pkl"))
    if not pats:
        sys.exit(f"{GEN} 底下沒有產出")

    print(f"{'組合':<46}{'條':>6}{'張':>4}{'相同%':>8}"
          f"{'走勢相關/水準':>10}")
    bad = 0
    for p in pats:
        tag = p.rsplit(os.sep, 3)[1]
        group = tag.rsplit("_", 2)[0]        # 去掉 _<variant>_seed<S>
        try:
            _, T, real = real_trace(group)
        except FileNotFoundError:
            print(f"{tag:<46}  找不到對應的訓練資料 {group}")
            bad += 1
            continue
        seqs, why = load_seqs(p, T)
        if seqs is None:
            print(f"{tag:<46}  {why}")
            bad += 1
            continue
        gen, same, corr, flat, ratio = stats(seqs, real)
        # burst 前後的平台期本來就有相鄰重複，20% 以下算正常；
        # 100% 才是條件沒生效。走勢平的（raw）看邊數水準而不是相關。
        bad_shape = ratio < 0.5 or ratio > 2.0 if flat else not (corr > 0.3)
        flag = "  <<<" if same > 20 or bad_shape else ""
        col = f"{ratio:>9.2f}x" if flat else f"{corr:>+10.3f}"
        print(f"{tag:<46}{len(seqs):>6}{len(seqs[0]):>4}"
              f"{same:>8.1f}{col}{flag}")
        if flag:
            bad += 1
    print(f"\n{len(pats)} 組，需要注意的 {bad} 組"
          "（相同比例超過 20%；走勢有起伏的看相關是否低於 0.3，平的看邊數水準是否偏離兩倍）")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        one(sys.argv[1])
    else:
        sweep()
