"""從 GraphMaker 的 log 抽出評估指標。

GraphMaker 不算 MMD/KS，它訓練一組判別器去區分真實圖與生成圖，
報告的是 ACC(G|G_hat) / ACC(G|G)——生成圖上訓練的分類器在真實圖上的表現，
除以真實圖上訓練的。越接近 1 表示生成圖越像真的。

另外兩項是 degree 分佈的 Pearson 與 Spearman 相關係數，也是越接近 1 越好。

log 裡混著 tqdm 的進度條，要逐行掃描而不是整段讀。
"""
import glob
import os
import re
import sys

DISCS = ["MLP", "SGC 1-layer", "SGC 2-layer", "GCN", "APPNP 1-layer",
         "APPNP 2-layer", "GAE 1-layer", "GAE 2-layer", "CN"]
ACC = re.compile(r"ACC\(G\|G_hat\) / ACC\(G\|G\):\s*([\d.eE+-]+)")
PEA = re.compile(r"Pearson correlation coefficient:\s*([\d.eE+-]+)")
SPE = re.compile(r"Spearman correlation coefficient:\s*([\d.eE+-]+)")


def parse(path):
    """回傳 {指標: 值}。判別器的名稱在數值的前一行。"""
    out = {}
    pending = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            # tqdm 用 \r 更新，一行可能塞了幾十次進度，只看最後一段
            line = raw.rstrip().split("\r")[-1].strip()
            if not line:
                continue

            for name in DISCS:
                if line.startswith(name + " discriminator"):
                    pending = name
                    break

            m = ACC.search(line)
            if m and pending:
                out[pending] = float(m.group(1))
                pending = None
                continue
            m = PEA.search(line)
            if m:
                out["Pearson"] = float(m.group(1))
                continue
            m = SPE.search(line)
            if m:
                out["Spearman"] = float(m.group(1))
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    pat = sys.argv[2] if len(sys.argv) > 2 else "*_seed*.log"
    paths = sorted(glob.glob(os.path.join(root, pat)))
    if not paths:
        raise SystemExit(f"{root} 底下找不到 log")

    cols = DISCS + ["Pearson", "Spearman"]
    print("組合|" + "|".join(cols))
    n_full = 0
    for p in paths:
        key = os.path.basename(p)[:-4]
        d = parse(p)
        if len(d) == len(cols):
            n_full += 1
        print(key + "|" + "|".join(
            f"{d[c]:.6f}" if c in d else "" for c in cols))

    print(f"# {len(paths)} 個檔案，{n_full} 個指標齊全", file=sys.stderr)


if __name__ == "__main__":
    main()
