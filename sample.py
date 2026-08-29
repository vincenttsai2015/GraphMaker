import os
import dgl
import networkx as nx
import torch
import torch.nn.functional as F

from huggingface_hub import hf_hub_download

from data import load_dataset, preprocess
from macro_data import MacroCondDataset, is_macro
from eval_utils import Evaluator
from setup_utils import set_seed

def main(args):
    if args.model_path is None:
        if args.dataset is None or args.type is None:
            raise ValueError("If model_path is not provided, both dataset and type must be specified for downloading a pre-trained model checkpoint.")
        
        filename = f"{args.dataset}_{args.type}.pth"
        
        print(f"Downloading pre-trained model: {filename}")
        args.model_path = hf_hub_download(repo_id="Graph-COM/GraphMaker", 
                                          filename=filename,
                                          cache_dir="./downloaded_cpts")
        print(f"Downloaded model to {args.model_path}")
    else:
        print(f"Loading local model from {args.model_path}")
    
    state_dict = torch.load(args.model_path)
    dataset = state_dict["dataset"]

    train_yaml_data = state_dict["train_yaml_data"]
    model_name = train_yaml_data["meta_data"]["variant"]

    print(f"Loaded GraphMaker-{model_name} model trained on {dataset}")
    print(f"Val Nll {state_dict['best_val_nll']}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    macro = MacroCondDataset(dataset) if is_macro(dataset) else None
    if macro is not None:
        # 條件由測試序列的觀測窗決定，沒有單一真實圖可餵給 Evaluator，
        # 指標一律由 eval_all_metrics.py 事後算。
        evaluator = None
        X_marginal = macro.X_marginal.to(device)
        Y_marginal = macro.Y_marginal.to(device)
        E_marginal = macro.E_marginal.to(device)
        num_nodes = macro.num_nodes
    else:
        g_real = load_dataset(dataset)
        X_one_hot_3d_real, Y_real, E_one_hot_real,\
            X_marginal, Y_marginal, E_marginal, X_cond_Y_marginals = preprocess(g_real)
        Y_one_hot_real = F.one_hot(Y_real)

        evaluator = Evaluator(dataset,
                              g_real,
                              X_one_hot_3d_real,
                              Y_one_hot_real)

        X_marginal = X_marginal.to(device)
        Y_marginal = Y_marginal.to(device)
        E_marginal = E_marginal.to(device)
        X_cond_Y_marginals = X_cond_Y_marginals.to(device)
        num_nodes = Y_real.size(0)

    if model_name == "Sync":
        from model import ModelSync

        model = ModelSync(X_marginal=X_marginal,
                          Y_marginal=Y_marginal,
                          E_marginal=E_marginal,
                          gnn_X_config=train_yaml_data["gnn_X"],
                          gnn_E_config=train_yaml_data["gnn_E"],
                          num_nodes=num_nodes,
                          **train_yaml_data["diffusion"]).to(device)

        model.graph_encoder.pred_X.load_state_dict(state_dict["pred_X_state_dict"])
        model.graph_encoder.pred_E.load_state_dict(state_dict["pred_E_state_dict"])

    elif model_name == "Async":
        from model import ModelAsync

        model = ModelAsync(X_marginal=X_marginal,
                           Y_marginal=Y_marginal,
                           E_marginal=E_marginal,
                           mlp_X_config=train_yaml_data["mlp_X"],
                           gnn_E_config=train_yaml_data["gnn_E"],
                           num_nodes=num_nodes,
                           **train_yaml_data["diffusion"]).to(device)

        model.graph_encoder.pred_X.load_state_dict(state_dict["pred_X_state_dict"])
        model.graph_encoder.pred_E.load_state_dict(state_dict["pred_E_state_dict"])

    model.eval()

    # Set seed for better reproducibility.
    set_seed()

    def _to_nx(E_0):
        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        s, d = E_0.nonzero().T
        g.add_edges_from(zip(s.tolist(), d.tolist()))
        return g

    seqs = []
    samples = []
    if macro is not None:
        # 每條測試序列依 t 生一張，條件是該序列觀測窗導出的 Y。
        n_seq = macro.test_Y.size(0)
        if args.max_seqs:
            n_seq = min(n_seq, args.max_seqs)
        print(f"測試序列 {n_seq} 條 x {macro.t_target} 張", flush=True)
        for s in range(n_seq):
            if s % 25 == 0:
                print(f"  取樣 {s}/{n_seq}", flush=True)
            seq = []
            for k in range(macro.t_target):
                # 一條測試序列要生 t_target 張，每張都重建一次 DataLoader。
                # 節點對只有一批，開子行程的成本會蓋過搬資料省下的時間。
                _, _, E_0 = model.sample(Y=macro.test_Y[s, k],
                                         num_workers=0)
                seq.append(_to_nx(E_0))
            seqs.append(seq)
    else:
        for _ in range(args.num_samples):
            X_0_one_hot, Y_0_one_hot, E_0 = model.sample()
            src, dst = E_0.nonzero().T
            g_sample = dgl.graph((src, dst), num_nodes=num_nodes).cpu()
            samples.append(g_sample)

            evaluator.add_sample(g_sample,
                                 X_0_one_hot.cpu(),
                                 Y_0_one_hot.cpu())

    # GM_GEN_DIR 給定時把生成圖存成 networkx 序列，與其他四個模型的
    # sampled_ts.pkl 同結構。
    _gen_dir = os.environ.get('GM_GEN_DIR', '')
    if _gen_dir and (seqs or samples):
        import pickle
        _tag = os.environ.get('GM_RUN_TAG', args.model_path and
                              os.path.basename(args.model_path)[:-4]
                              or 'sample')
        if seqs:
            _seqs = seqs
            _note = f'{len(_seqs)} 條 x {len(_seqs[0])} 張'
        else:
            # 無條件生成沒有時間軸，每張複製 GM_SEQ_LEN 次當成不變的序列。
            _T = int(os.environ.get('GM_SEQ_LEN', '16'))
            _seqs = []
            for _g in samples:
                _nx = nx.Graph()
                _nx.add_nodes_from(range(_g.num_nodes()))
                _u, _v = _g.edges()
                _nx.add_edges_from(zip(_u.tolist(), _v.tolist()))
                _seqs.append([_nx.copy() for _ in range(_T)])
            _note = f'{len(_seqs)} 張 x 複製 {_T} 次'
        _d = os.path.join(_gen_dir, _tag, 'GraphMaker')
        os.makedirs(_d, exist_ok=True)
        with open(os.path.join(_d, 'sampled_ts.pkl'), 'wb') as _f:
            pickle.dump(_seqs, _f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'生成圖已存到 {_d}：{_note}', flush=True)

    if seqs:
        import numpy as _np
        _per_t = [_np.mean([len(s[k].edges()) for s in seqs])
                  for k in range(len(seqs[0]))]
        print('每張平均邊數 t=' + str(macro.t_split) + '..:',
              ' '.join(f'{v:.0f}' for v in _per_t), flush=True)

    if evaluator is not None:
        evaluator.summary()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--dataset", type=str,
                        help="Dataset name. Only specify it if you want to use a pre-trained model.")
    parser.add_argument("--type", type=str, choices=["sync", "async"],
                        help="Model type. Only specify it if you want to use a pre-trained model.")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate.")
    parser.add_argument("--max_seqs", type=int, default=0,
                        help="條件式資料只生前 N 條測試序列，0 表示全部。")
    args = parser.parse_args()

    main(args)
