import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from ..utils.utils import debug_print
from pykt.config import que_type_models
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_loss(model, ys, r, rshft, sm, preloss=[], s_opts_shft=None):
    model_name = model.model_name

    if model_name in ["atdkt", "simplekt", "bakt_time", "sparsekt", "fliicbsimplekt", "simplekt_enhance_pro", "simplekt_enhance_pro_qid", "simplekt_qid"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # print(f"loss1: {y.shape}")
        loss1 = binary_cross_entropy(y.double(), t.double())

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            else:
                loss = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1*loss1+model.l2*ys[1]
        else:
            loss = loss1

    elif model_name in ["rkt","dimkt","dkt", "dkt_forget", "dkvmn","deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes"]:
        # 转成1维的张量，然后算二元交叉熵
        y = torch.masked_select(ys[0], sm)
        # print(f"y.shape: {y.shape}")
        t = torch.masked_select(rshft, sm)
        # print(f"t.shape: {t.shape}")
        loss = binary_cross_entropy(y.double(), t.double())

    elif model_name in ["dkt_otkt"]:
        # 获取有效位置的索引
        valid_indices = sm.nonzero(as_tuple=True)
        batch_idx, seq_idx = valid_indices
        
        # KT损失计算 - 预测学生是否回答正确
        y_kt = torch.masked_select(ys[0], sm)
        t_kt = torch.masked_select(rshft, sm)
        kt_loss = binary_cross_entropy(y_kt.double(), t_kt.double())
        
        # OT损失计算 - 预测学生会选择哪个选项
        # ys[1]形状为[batch_size, seq_len, max_opt]，表示每个选项的预测概率
        ot_probs = ys[1][batch_idx, seq_idx, :]  # [num_valid, max_opt]
        student_choices = s_opts_shft[batch_idx, seq_idx]  # [num_valid]，学生实际选择的选项
        
        # 计算交叉熵损失 - 预测学生选择
        ot_loss = cross_entropy(ot_probs, student_choices.long())
        
        # 使用λ参数平衡两个任务的损失
        lambda_kt = getattr(model, 'lambda_kt', 0.5)
        loss = lambda_kt * kt_loss + (1 - lambda_kt) * ot_loss

    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y_next.double(), r_next.double())

        loss_r = binary_cross_entropy(y_curr.double(), r_curr.double()) # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
    elif model_name in ["akt", "cakt", "folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx", "dtransformer", "akt_enhance_pro", "akt_enhance_pro_qid", "akt_qid"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        criterion = nn.BCELoss(reduction='none')        
        loss = criterion(y, t).sum()
    elif model_name in ['lightkt', "expertkt", "pointkt", 'dkt_enhance_pro', "sakt_pro", "sakt_enhance_pro", "dkvmn_pro", "dkvmn_enhance_pro"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # criterion = nn.BCELoss(reduction='none') 
        # loss = criterion(y,t).sum()
        loss = binary_cross_entropy(y.double(), t.double())
    # 在cal_loss函数中添加这个elif分支，放在其他elif之后

    # ===== 优化后的 dkt_kt_ot 损失计算 =====
    elif model_name in ["dkt_kt_ot"]:
        # 获取有效位置的索引
        valid_indices = sm.nonzero(as_tuple=True) 
        batch_idx, seq_idx = valid_indices
        
        # KT损失计算 - 预测学生是否回答正确
        y_kt = torch.masked_select(ys[0], sm)
        t_kt = torch.masked_select(rshft, sm)
        kt_loss = binary_cross_entropy(y_kt.double(), t_kt.double())
        
        # OT损失计算 - 预测学生会选择哪个选项
        if len(ys) > 1 and s_opts_shft is not None:
            # ===== 优化版本：消除for循环 =====
            # ys[1]形状为[batch_size, seq_len, max_opt]，已经通过mask处理了无效选项
            ot_probs = ys[1][batch_idx, seq_idx, :]  # [num_valid, max_opt]
            student_choices = s_opts_shft[batch_idx, seq_idx].long()  # [num_valid]
            
            # 创建有效选项mask，过滤掉无效的学生选择
            valid_choice_mask = (student_choices >= 0) & (student_choices < ot_probs.shape[-1])
            
            if valid_choice_mask.any():
                # 只对有效选择计算损失
                valid_ot_probs = ot_probs[valid_choice_mask]  # [num_valid_choices, max_opt]
                valid_student_choices = student_choices[valid_choice_mask]  # [num_valid_choices]
                
                # 直接使用cross_entropy，因为无效选项已经被mask掉了（logits为-1e9）
                ot_loss = cross_entropy(valid_ot_probs, valid_student_choices)
            else:
                ot_loss = torch.tensor(0.0, device=ys[0].device)
        else:
            ot_loss = torch.tensor(0.0, device=ys[0].device)
        
        # 使用λ参数平衡两个任务的损失
        lambda_kt = getattr(model, 'lambda_kt', 0.7)
        loss = lambda_kt * kt_loss + (1 - lambda_kt) * ot_loss

    return loss


def model_forward(model, data, opt, emb_size=None, dataset_name=None, step_size=None, step_m=None, grad_clip=None, mm=None, rel=None):
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    if model_name in ["dkt_forget", "bakt_time"]:
        dcur, dgaps = data
    else:
        dcur = data
        # print(f"dcur: {dcur}")
    if model_name in ["dimkt"]:
        q, c, r, t,sd,qd = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device),dcur["sdseqs"].to(device),dcur["qdseqs"].to(device)
        qshft, cshft, rshft, tshft,sdshft,qdshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device),dcur["shft_sdseqs"].to(device),dcur["shft_qdseqs"].to(device)
    else:
        q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)

    if model_name in ["dkt_otkt"]:
        s_opts, a_opts = dcur["student_opts"].to(device), dcur["correct_opts"].to(device)
        s_opts_shft, a_opts_shft = dcur["shft_student_opts"].to(device), dcur["shft_correct_opts"].to(device)


    # # print(dcur["student_opts"])
    # # print(dcur["correct_options"])
    # print("+"*100)
    # print(dcur.keys())
    # print("+"*100)    

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:,0:1], tshft), dim=1)
    elif model_name in ["rkt"]:
        y, attn = model(dcur, rel, train=True)
        ys.append(y[:,1:])
    if model_name in ["atdkt"]:
        # is_repeat = dcur["is_repeat"]
        y, y2, y3 = model(dcur, train=True)
        if model.emb_type.find("bkt") == -1 and model.emb_type.find("addcshft") == -1:
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # y2 = (y2 * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3] # first: yshft
    elif model_name in ["simplekt", "sparsekt", "fliicbsimplekt", "simplekt_enhance_pro", "simplekt_enhance_pro_qid", "simplekt_qid"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["dtransformer"]:
        if model.emb_type == "qid_cl":
            y, loss = model.get_cl_loss(cc.long(), cr.long(), cq.long())  # with cl loss
        else:
            y, loss = model.get_loss(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(loss)
    elif model_name in ["bakt_time"]:
        y, y2, y3 = model(dcur, dgaps, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:,0:1], dcur["shft_itseqs"]), dim=1)
    if model_name in ["dkt"]:
        y = model(c.long(), r.long())  
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # print(f"y.shape: {y.shape}")
        ys.append(y) # first: yshft
    elif model_name in ["dkt_otkt"]:
        # kt [batch_size, seq_len, num_c]
        # ot [batch_size, seq_len, max_opt]

        kt, ot = model(q.long(), c.long(), r.long(), s_opts.long())  

        # 计算kt任务最后的结果
        kt = (kt * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(kt)
        
        # 计算ot最后的结果，但是由于ot本身就已经转成最后的结果了，所以不需要再转
        ys.append(ot)

        # 计算损失，传递学生选择的选项
        loss = cal_loss(model, ys, r, rshft, sm, preloss, s_opts_shft=s_opts_shft)
        
        return loss
    # model_forward函数中dkt_kt_ot模型的处理部分（替换原有的elif分支）

    elif model_name in ["dkt_kt_ot"]:
        # 检查是否有选项数据
        if "student_opts" in dcur and "correct_opts" in dcur:
            s_opts, c_opts = dcur["student_opts"].to(device), dcur["correct_opts"].to(device)
            s_opts_shft, c_opts_shft = dcur["shft_student_opts"].to(device), dcur["shft_correct_opts"].to(device)

            # 打印调试信息（可选，生产环境请注释掉）
            # print("+"*50)
            # print(f"s_opts.shape: {s_opts.shape}")
            # print(f"c_opts.shape: {c_opts.shape}")
            # print(f"s_opts_shft.shape: {s_opts_shft.shape}")
            # print(f"c_opts_shft.shape: {c_opts_shft.shape}")
            # print("+"*50)
            
            # 调用优化后的多任务模型，传入选项信息
            kt_pred, ot_pred = model(
                last_pro=q.long(), 
                last_ans=r.long(), 
                last_skill=c.long(), 
                next_pro=qshft.long(), 
                next_skill=cshft.long(), 
                student_opts=s_opts.long(), 
                correct_opts=c_opts.long(),
                next_student_opts=s_opts_shft.long()
            )
            ys = [kt_pred, ot_pred]
            
            # 计算损失，传递学生选择的选项
            loss = cal_loss(model, ys, r, rshft, sm, preloss, s_opts_shft=s_opts_shft)
            
        else:
            # 如果没有选项数据，只进行KT任务
            kt_pred, _ = model(
                last_pro=q.long(), 
                last_ans=r.long(), 
                last_skill=c.long(), 
                next_pro=qshft.long(), 
                next_skill=cshft.long()
            )
            ys = [kt_pred]
            loss = cal_loss(model, ys, r, rshft, sm, preloss)
        
        return loss
    elif model_name in ['dkt_enhance_pro', "sakt_pro", "sakt_enhance_pro"]:
        y = model(q.long(), r.long(), c.long(), qshft.long(), cshft.long())
        ys.append(y)
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
        return loss
    elif model_name in ['dkvmn_pro', 'dkvmn_enhance_pro']:
        y = model(cq.long(), cr.long())
        ys.append(y[:,1:])  # ys.append(y[:,1:])
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
        return loss 
    elif model_name in ["expertkt"]:  
        # def forward(self, last_pro, last_ans, last_skill, next_pro, next_skill):
        grad_clip = 15.0
        y, preloss = model(q.long(), r.long(), c.long(), qshft.long(), cshft.long())
        ys.append(y)
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
        
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        return loss
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn","deep_irt", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:,1:])
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name in ["akt", "cakt", "folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx", "akt_enhance_pro", "akt_enhance_pro_qid", "akt_qid"]:               
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm)
        # at
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        pred_res, _ = model(c.long(), r.long(), p_adv)
        # second loss
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm)
        loss = loss + model.beta * adv_loss
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)  
    # cal loss
    elif model_name == "lpkt":
        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])  
    elif model_name == "hawkes":
        # ct = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        # csm = torch.cat((dcur["smasks"][:,0:1], dcur["smasks"]), dim=1)
        # y = model(cc[0:1,0:5].long(), cq[0:1,0:5].long(), ct[0:1,0:5].long(), cr[0:1,0:5].long(), csm[0:1,0:5].long())
        y = model(cc.long(), cq.long(), ct.long(), cr.long())#, csm.long())
        ys.append(y[:, 1:])
    elif model_name in que_type_models and model_name not in ["lpkt", "rkt", "simplekt_enhance_pro_qid", "simplekt_qid"]:
        y,loss = model.train_one_step(data)
    elif model_name == "dimkt":
        y = model(q.long(),c.long(),sd.long(),qd.long(),r.long(),qshft.long(),cshft.long(),sdshft.long(),qdshft.long())
        ys.append(y) 

    if model_name not in ["atkt", "atktfix"]+que_type_models or model_name in ["lpkt", "rkt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    if model_name in ["simplekt_enhance_pro_qid", "simplekt_qid", "akt_enhance_pro_qid", "akt_qid"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    return loss
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, emb_size=128, dataset_name="assist2009", step_size=None, step_m=None, grad_clip=None, mm=None, data_config=None, fold=None):
    max_auc, best_epoch = 0, -1
    train_step = 0

    rel = None
    if model.model_name == "rkt":
        # print(f"data_config: {data_config}")
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))

    if model.model_name=='lpkt':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step+=1
            if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt",  "lightkt", "expertkt", "pointkt", "dkt_enhance_pro", "dkt_kt_ot", "sakt_pro", "sakt_enhance_pro", "dkvmn_pro", "dkvmn_enhance_pro", "simplekt_enhance_pro_qid", "simplekt_qid", "akt_enhance_pro_qid", "akt_qid"]:
                model.model.train()
            else:
                model.train()
            if model.model_name=='rkt':
                loss = model_forward(model, data, rel)
            else:
                loss = model_forward(model, data, opt, emb_size, dataset_name, step_size, step_m, grad_clip, mm)
                # loss = model_forward(model, data, opt)
            if model.model_name not in ["lightkt", "expertkt", "pointkt"]:
                opt.zero_grad()
                loss.backward()#compute gradients
            if model.model_name == "rkt":
                clip_grad_norm_(model.parameters(), model.grad_clip)
            if model.model_name == "dtransformer":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if model.model_name not in ["lightkt", "expertkt", "pointkt"]:
                opt.step()#update model’s parameters
            loss_mean.append(loss.detach().cpu().numpy())
            if model.model_name == "gkt" and train_step%10==0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text = text,fuc_name="train_model")
        if model.model_name=='lpkt':
            scheduler.step()#update each epoch
        loss_mean = np.mean(loss_mean)
        
        if model.model_name=='rkt':
            auc, acc = evaluate(model, valid_loader, model.model_name, rel)
        else:
            auc, acc = evaluate(model, valid_loader, model.model_name)

        ### atkt 有diff， 以下代码导致的
        ### auc, acc = round(auc, 4), round(acc, 4)

        if auc > max_auc+1e-3:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            validauc, validacc = auc, acc
        print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")


        if i - best_epoch >= 10:
            break
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch
