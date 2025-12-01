import torch
import numpy as np
import os

from .dkt import DKT
from .dkt_enhance_pro import DKT_Enhance_Pro
from .sakt_pro import SAKT_PRO
from .sakt_enhance_pro import SAKT_Enhance_PRO
from .dkvmn_pro import DKVMN_PRO
from .dkvmn_enhance_pro import DKVMN_Enhance_PRO
from .simplekt_enhance_pro_qid import simpleKT_enhance_pro_qid
from .simplekt_qid import simpleKT_qid
from .akt_enhance_pro_qid import AKT_Enhance_Pro_qid
from .akt_qid import AKT_qid

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_model(model_name, model_config, data_config, emb_type, dataset_name=None):
    # print("+"*100)
    # print(f"dataset_name is {dataset_name}")
    # print("+"*100)
    if model_name == "dkt":
        model = DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_otkt":
        model = DKT_otkt(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_abqr_kc":
        model = DKT_ABQR_KC(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "deep_irt":
        model = DeepIRT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt":
        model = SAKT(data_config["num_c"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt_pro":
        model = SAKT_PRO(data_config["num_q"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt_enhance_pro":
        model = SAKT_Enhance_PRO(data_config["num_q"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "saint":
        model = SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_forget":
        model = DKTForget(data_config["num_c"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config).to(device)
    elif model_name == "akt":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "akt_enhance_pro":
        model = AKT_Enhance_Pro(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "akt_enhance_pro_qid":
        model = AKT_Enhance_Pro_qid(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "akt_qid":
        model = AKT_qid(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "cakt":
        model = CAKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "folibikt":
        model = folibiKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "kqn":
        model = KQN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "atkt":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=False).to(device)
    elif model_name == "atktfix":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=True).to(device)
    elif model_name == "gkt":
        graph_type = model_config['graph_type']
        fname = f"gkt_graph_{graph_type}.npz"
        graph_path = os.path.join(data_config["dpath"], fname)
        if os.path.exists(graph_path):
            graph = torch.tensor(np.load(graph_path, allow_pickle=True)['matrix']).float()
        else:
            graph = get_gkt_graph(data_config["num_c"], data_config["dpath"], 
                    data_config["train_valid_original_file"], data_config["test_original_file"], graph_type=graph_type, tofile=fname)
            graph = torch.tensor(graph).float()
        model = GKT(data_config["num_c"], **model_config,graph=graph,emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "lpkt":
        qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
        if os.path.exists(qmatrix_path):
            q_matrix = np.load(qmatrix_path, allow_pickle=True)['matrix']
        else:
            q_matrix = generate_qmatrix(data_config)
        q_matrix = torch.tensor(q_matrix).float().to(device)
        model = LPKT(data_config["num_at"], data_config["num_it"], data_config["num_q"], data_config["num_c"], **model_config, q_matrix=q_matrix, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_abqr":
        model = DKT_ABQR(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"])
    elif model_name == "dkt_enhance_pro":
        model = DKT_Enhance_Pro(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name)
    elif model_name == "dkvmn_pro":
        model = DKVMN_PRO(data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkvmn_enhance_pro":
        model = DKVMN_Enhance_PRO(data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "dkt_kt_ot":
        model = DKT_KT_OT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"])
    elif model_name == "expertkt":
        model = ExpertKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"])
    elif model_name == "lightkt":
        model = LightKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"])
    elif model_name == "pointkt":
        model = PointKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"])
    elif model_name == "skvmn":
        model = SKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)   
    elif model_name == "hawkes":
        if data_config["num_q"] == 0 or data_config["num_c"] == 0:
            print(f"model: {model_name} needs questions ans concepts! but the dataset has no both")
            return None
        model = HawkesKT(data_config["num_c"], data_config["num_q"], **model_config)
        model = model.double()
        # print("===before init weights"+"@"*100)
        # model.printparams()
        model.apply(model.init_weights)
        # print("===after init weights")
        # model.printparams()
        model = model.to(device)
    elif model_name == "iekt":
        model = IEKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device, dataset_name=dataset_name).to(device)   
    elif model_name == "qdkt":
        model = QDKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "qdkt_enhance_pro":
        model = QDKT_Enhance_PRO(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device, dataset_name=dataset_name).to(device)
    elif model_name == "qikt":
        model = QIKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device, dataset_name=dataset_name).to(device)
    elif model_name == "atdkt":
        model = ATDKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "bakt_time":
        model = BAKTTime(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "simplekt":
        model = simpleKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "simplekt_abqr":
        model = simpleKT_ABQR(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "simplekt_enhance_pro":
        model = simpleKT_enhance_pro(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "simplekt_enhance_pro_qid":
        model = simpleKT_enhance_pro_qid(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "simplekt_qid":
        model = simpleKT_qid(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], dataset_name=dataset_name).to(device)
    elif model_name == "fliicbsimplekt":
        model = fliicbsimpleKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dimkt":
        model = DIMKT(data_config["num_q"],data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sparsekt":
        model = sparseKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "rkt":
        model = RKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)    
    elif model_name == "dtransformer":
        model = DTransformer(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type,
                     emb_path=data_config["emb_path"]).to(device)      
    else:
        print("The wrong model name was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path, dataset_name=None):
    model = init_model(model_name, model_config, data_config, emb_type, dataset_name)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    model.load_state_dict(net)
    return model