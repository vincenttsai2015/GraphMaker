import os
import dgl
import torch
import torch.nn.functional as F

from huggingface_hub import hf_hub_download

from data import load_dataset, preprocess
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

    samples = []
    for _ in range(args.num_samples):
        X_0_one_hot, Y_0_one_hot, E_0 = model.sample()
        src, dst = E_0.nonzero().T
        g_sample = dgl.graph((src, dst), num_nodes=num_nodes).cpu()
        samples.append(g_sample)

        evaluator.add_sample(g_sample,
                             X_0_one_hot.cpu(),
                             Y_0_one_hot.cpu())

    # GM_GEN_DIR 給定時把生成圖存成 networkx 序列，與其他四個模型的
    # sampled_ts.pkl 同結構。GraphMaker 生成的是單一靜態圖，沒有時間軸，
    # 所以每張複製 GM_SEQ_LEN 次當成「完全不變的序列」——那些時序指標
    # 因此量到的是零動態的基準線，不是模型能力。
    _gen_dir = os.environ.get('GM_GEN_DIR', '')
    if _gen_dir and samples:
        import pickle
        import networkx as nx
        _T = int(os.environ.get('GM_SEQ_LEN', '32'))
        _tag = os.environ.get('GM_RUN_TAG', args.model_path and
                              os.path.basename(args.model_path)[:-4]
                              or 'sample')
        _seqs = []
        for _g in samples:
            _nx = nx.Graph()
            _nx.add_nodes_from(range(_g.num_nodes()))
            _u, _v = _g.edges()
            _nx.add_edges_from(zip(_u.tolist(), _v.tolist()))
            _seqs.append([_nx.copy() for _ in range(_T)])
        _d = os.path.join(_gen_dir, _tag, 'GraphMaker')
        os.makedirs(_d, exist_ok=True)
        with open(os.path.join(_d, 'sampled_ts.pkl'), 'wb') as _f:
            pickle.dump(_seqs, _f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'生成圖已存到 {_d}：{len(_seqs)} 張 x 複製 {_T} 次', flush=True)

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
    args = parser.parse_args()

    main(args)
