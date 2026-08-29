"""注入巨觀動態的條件式資料集。

`npz_to_graphmaker.py` 產出的每一筆是目標窗的一張 snapshot，條件編在 `Y`：

    Y = 觀測窗活躍分箱 * t_target + (t - t_split)

`Y` 在 GraphMaker 裡是純條件——擴散不碰、loss 不預測、reverse chain 每一步
原樣餵入，所以把觀測窗行為與目標時刻放進去就等於條件化，不必改網路。

節點數固定（補位到 num_nodes），因此所有圖共用同一組 edge_index。
邊界分佈取全體訓練圖，不是單張。
"""
import os
import pickle

import torch
import torch.nn.functional as F


class MacroCondDataset:
    def __init__(self, data_name, root=None):
        root = root or os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "data", "macro")
        with open(os.path.join(root, f"{data_name}.pkl"), "rb") as f:
            d = pickle.load(f)

        self.name = data_name
        self.num_nodes = int(d["num_nodes"])
        self.num_attrs = int(d["num_attrs"])
        self.num_classes_Y = int(d["num_classes_Y"])
        self.t_split = int(d["t_split"])
        self.t_target = int(d["t_target"])

        self.train = d["train"]
        self.test_Y = torch.from_numpy(d["test_Y"]).long()

        self.X_marginal = torch.tensor(d["X_marginal"],
                                       dtype=torch.float32).reshape(
                                           self.num_attrs, 2)
        self.E_marginal = torch.tensor(d["E_marginal"], dtype=torch.float32)
        self.Y_marginal = torch.tensor(d["Y_marginal"], dtype=torch.float32)

        self.dataset = d["dataset"]
        self.mode = d["mode"]
        self.layer_name = d["layer_name"]

    def __len__(self):
        return len(self.train)

    def tensors(self, i, device):
        """第 i 張訓練圖的 (X_one_hot_3d, E_one_hot, Y)。

        Returns
        -------
        X_one_hot_3d : (F, N, 2)
        E_one_hot : (N, N, 2)
        Y : (N)
        """
        g = self.train[i]
        n = self.num_nodes

        X = torch.from_numpy(g["X"]).long().to(device)          # (N)
        X_one_hot_3d = F.one_hot(X, num_classes=2).float().unsqueeze(0)

        E = torch.zeros(n, n, device=device)
        if len(g["src"]):
            src = torch.from_numpy(g["src"]).long().to(device)
            dst = torch.from_numpy(g["dst"]).long().to(device)
            E[src, dst] = 1.
            E[dst, src] = 1.
        E_one_hot = F.one_hot(E.long(), num_classes=2).float()

        Y = torch.from_numpy(g["Y"]).long().to(device)
        return X_one_hot_3d, E_one_hot, Y


def is_macro(data_name):
    return data_name.startswith("macro_")
