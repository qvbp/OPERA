# import os
# import sys
# import pandas as pd
# import numpy as np
# import json
# import copy

# ALL_KEYS = ["fold", "uid", "questions", "concepts", "responses", "timestamps",
#             "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow", "cidxs"]
# ONE_KEYS = ["fold", "uid"]


# def read_data(fname, min_seq_len=3, response_set=[0, 1]):
#     effective_keys = set()
#     dres = dict()
#     delstu, delnum, badr = 0, 0, 0
#     goodnum = 0
#     with open(fname, "r", encoding="utf8") as fin:
#         i = 0
#         lines = fin.readlines()
#         dcur = dict()
#         while i < len(lines):
#             line = lines[i].strip()
#             if i % 6 == 0:  # stuid
#                 effective_keys.add("uid")
#                 tmps = line.split(",")
#                 if "(" in tmps[0]:
#                     stuid, seq_len = tmps[0].replace('(', ''), int(tmps[2])
#                 else:
#                     stuid, seq_len = tmps[0], int(tmps[1])
#                 if seq_len < min_seq_len:  # delete use seq len less than min_seq_len
#                     i += 6
#                     dcur = dict()
#                     delstu += 1
#                     delnum += seq_len
#                     continue
#                 dcur["uid"] = stuid
#                 goodnum += seq_len
#             # elif i % 6 == 1:  # question ids / names
#             #     qs = []
#             #     if line.find("NA") == -1:
#             #         effective_keys.add("questions")
#             #         qs = line.split(",")
#             #     dcur["questions"] = qs
#             elif i % 6 == 1:  # question ids / names
#                 qs = []
#                 # if line.find("NA") == -1:
#                 if "NA" not in line.split(","):
#                     effective_keys.add("questions")
#                     qs = line.split(",")
                    
#                 dcur["questions"] = qs
#             elif i % 6 == 2:  # concept ids / names
#                 cs = []
#                 if line.find("NA") == -1:
#                     effective_keys.add("concepts")
#                     cs = line.split(",")
#                 dcur["concepts"] = cs
#             elif i % 6 == 3:  # responses
#                 effective_keys.add("responses")
#                 rs = []
#                 if line.find("NA") == -1:
#                     flag = True
#                     for r in line.split(","):
#                         try:
#                             r = int(r)
#                             if r not in response_set:  # check if r in response set.
#                                 print(f"error response in line: {i}")
#                                 flag = False
#                                 break
#                             rs.append(r)
#                         except:
#                             print(f"error response in line: {i}")
#                             flag = False
#                             break
#                     if not flag:
#                         i += 3
#                         dcur = dict()
#                         badr += 1
#                         continue
#                 dcur["responses"] = rs
#             elif i % 6 == 4:  # timestamps
#                 ts = []
#                 if line.find("NA") == -1:
#                     effective_keys.add("timestamps")
#                     ts = line.split(",")
#                 dcur["timestamps"] = ts
#             elif i % 6 == 5:  # usets
#                 usets = []
#                 if line.find("NA") == -1:
#                     effective_keys.add("usetimes")
#                     usets = line.split(",")
#                 dcur["usetimes"] = usets

#                 for key in effective_keys:
#                     dres.setdefault(key, [])
#                     if key != "uid":
#                         dres[key].append(",".join([str(k) for k in dcur[key]]))
#                     else:
#                         dres[key].append(dcur[key])
#                 dcur = dict()
#             i += 1
#     df = pd.DataFrame(dres)
#     print(
#         f"delete bad stu num of len: {delstu}, delete interactions: {delnum}, of r: {badr}, good num: {goodnum}")
#     return df, effective_keys


# def extend_multi_concepts(df, effective_keys):
#     if "questions" not in effective_keys or "concepts" not in effective_keys:
#         print("has no questions or concepts! return original.")
#         return df, effective_keys
#     extend_keys = set(df.columns) - {"uid"}

#     dres = {"uid": df["uid"]}
#     for _, row in df.iterrows():
#         dextend_infos = dict()
#         for key in extend_keys:
#             dextend_infos[key] = row[key].split(",")
#         dextend_res = dict()
#         for i in range(len(dextend_infos["questions"])):
#             dextend_res.setdefault("is_repeat", [])
#             if dextend_infos["concepts"][i].find("_") != -1:
#                 ids = dextend_infos["concepts"][i].split("_")
#                 dextend_res.setdefault("concepts", [])
#                 dextend_res["concepts"].extend(ids)
#                 for key in extend_keys:
#                     if key != "concepts":
#                         dextend_res.setdefault(key, [])
#                         dextend_res[key].extend(
#                             [dextend_infos[key][i]] * len(ids))
#                 dextend_res["is_repeat"].extend(
#                     ["0"] + ["1"] * (len(ids) - 1))  # 1: repeat, 0: original
#             else:
#                 for key in extend_keys:
#                     dextend_res.setdefault(key, [])
#                     dextend_res[key].append(dextend_infos[key][i])
#                 dextend_res["is_repeat"].append("0")
#         for key in dextend_res:
#             dres.setdefault(key, [])
#             dres[key].append(",".join(dextend_res[key]))

#     finaldf = pd.DataFrame(dres)
#     effective_keys.add("is_repeat")
#     return finaldf, effective_keys


# def id_mapping(df):
#     id_keys = ["questions", "concepts", "uid"]
#     dres = dict()
#     dkeyid2idx = dict()
#     print(f"df.columns: {df.columns}")
#     for key in df.columns:
#         if key not in id_keys:
#             dres[key] = df[key]
#     for i, row in df.iterrows():
#         for key in id_keys:
#             if key not in df.columns:
#                 continue
#             dkeyid2idx.setdefault(key, dict())
#             dres.setdefault(key, [])
#             curids = []
#             for id in row[key].split(","):
#                 if id not in dkeyid2idx[key]:
#                     dkeyid2idx[key][id] = len(dkeyid2idx[key])
#                 curids.append(str(dkeyid2idx[key][id]))
#             dres[key].append(",".join(curids))
#     finaldf = pd.DataFrame(dres)
#     return finaldf, dkeyid2idx


# def train_test_split(df, test_ratio=0.2):
#     df = df.sample(frac=1.0, random_state=1024)
#     datanum = df.shape[0]
#     test_num = int(datanum * test_ratio)
#     train_num = datanum - test_num
#     train_df = df[0:train_num]
#     test_df = df[train_num:]
#     # report
#     print(
#         f"total num: {datanum}, train+valid num: {train_num}, test num: {test_num}")
#     return train_df, test_df


# def KFold_split(df, k=5):
#     df = df.sample(frac=1.0, random_state=1024)
#     datanum = df.shape[0]
#     test_ratio = 1 / k
#     test_num = int(datanum * test_ratio)
#     rest = datanum % k

#     start = 0
#     folds = []
#     for i in range(0, k):
#         if rest > 0:
#             end = start + test_num + 1
#             rest -= 1
#         else:
#             end = start + test_num
#         folds.extend([i] * (end - start))
#         print(f"fold: {i+1}, start: {start}, end: {end}, total num: {datanum}")
#         start = end
#     # report
#     finaldf = copy.deepcopy(df)
#     finaldf["fold"] = folds
#     return finaldf


# def save_dcur(row, effective_keys):
#     dcur = dict()
#     for key in effective_keys:
#         if key not in ONE_KEYS:
#             # [int(i) for i in row[key].split(",")]
#             dcur[key] = row[key].split(",")
#         else:
#             dcur[key] = row[key]
#     return dcur


# def generate_sequences(df, effective_keys, min_seq_len=3, maxlen=200, pad_val=-1):
#     save_keys = list(effective_keys) + ["selectmasks"]
#     dres = {"selectmasks": []}
#     dropnum = 0
#     for i, row in df.iterrows():
#         dcur = save_dcur(row, effective_keys)

#         rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
#         j = 0
#         while lenrs >= j + maxlen:
#             rest = rest - (maxlen)
#             for key in effective_keys:
#                 dres.setdefault(key, [])
#                 if key not in ONE_KEYS:
#                     # [str(k) for k in dcur[key][j: j + maxlen]]))
#                     dres[key].append(",".join(dcur[key][j: j + maxlen]))
#                 else:
#                     dres[key].append(dcur[key])
#             dres["selectmasks"].append(",".join(["1"] * maxlen))

#             j += maxlen
#         if rest < min_seq_len:  # delete sequence len less than min_seq_len
#             dropnum += rest
#             continue

#         pad_dim = maxlen - rest
#         for key in effective_keys:
#             dres.setdefault(key, [])
#             if key not in ONE_KEYS:
#                 paded_info = np.concatenate(
#                     [dcur[key][j:], np.array([pad_val] * pad_dim)])
#                 dres[key].append(",".join([str(k) for k in paded_info]))
#             else:
#                 dres[key].append(dcur[key])
#         dres["selectmasks"].append(
#             ",".join(["1"] * rest + [str(pad_val)] * pad_dim))

#     # after preprocess data, report
#     dfinal = dict()
#     for key in ALL_KEYS:
#         if key in save_keys:
#             dfinal[key] = dres[key]
#     finaldf = pd.DataFrame(dfinal)
#     print(f"dropnum: {dropnum}")
#     return finaldf


# def generate_window_sequences(df, effective_keys, maxlen=200, pad_val=-1):
#     save_keys = list(effective_keys) + ["selectmasks"]
#     dres = {"selectmasks": []}
#     for i, row in df.iterrows():
#         dcur = save_dcur(row, effective_keys)
#         lenrs = len(dcur["responses"])
#         if lenrs > maxlen:
#             for key in effective_keys:
#                 dres.setdefault(key, [])
#                 if key not in ONE_KEYS:
#                     # [str(k) for k in dcur[key][0: maxlen]]))
#                     dres[key].append(",".join(dcur[key][0: maxlen]))
#                 else:
#                     dres[key].append(dcur[key])
#             dres["selectmasks"].append(",".join(["1"] * maxlen))
#             for j in range(maxlen+1, lenrs+1):
#                 for key in effective_keys:
#                     dres.setdefault(key, [])
#                     if key not in ONE_KEYS:
#                         dres[key].append(",".join([str(k)
#                                          for k in dcur[key][j-maxlen: j]]))
#                     else:
#                         dres[key].append(dcur[key])
#                 dres["selectmasks"].append(
#                     ",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
#         else:
#             for key in effective_keys:
#                 dres.setdefault(key, [])
#                 if key not in ONE_KEYS:
#                     pad_dim = maxlen - lenrs
#                     paded_info = np.concatenate(
#                         [dcur[key][0:], np.array([pad_val] * pad_dim)])
#                     dres[key].append(",".join([str(k) for k in paded_info]))
#                 else:
#                     dres[key].append(dcur[key])
#             dres["selectmasks"].append(
#                 ",".join(["1"] * lenrs + [str(pad_val)] * pad_dim))

#     dfinal = dict()
#     for key in ALL_KEYS:
#         if key in save_keys:
#             # print(f"key: {key}, len: {len(dres[key])}")
#             dfinal[key] = dres[key]
#     finaldf = pd.DataFrame(dfinal)
#     return finaldf


# def get_inter_qidx(df):
#     """add global id for each interaction"""
#     qidx_ids = []
#     bias = 0
#     inter_num = 0
#     for _, row in df.iterrows():
#         ids_list = [str(x+bias)
#                     for x in range(len(row['responses'].split(',')))]
#         inter_num += len(ids_list)
#         ids = ",".join(ids_list)
#         qidx_ids.append(ids)
#         bias += len(ids_list)
#     assert inter_num-1 == int(ids_list[-1])

#     return qidx_ids


# def add_qidx(dcur, global_qidx):
#     idxs, rests = [], []
#     # idx = -1
#     for r in dcur["is_repeat"]:
#         if str(r) == "0":
#             global_qidx += 1
#         idxs.append(global_qidx)
#     # print(dcur["is_repeat"])
#     # print(f"idxs: {idxs}")
#     # print("="*20)
#     for i in range(0, len(idxs)):
#         rests.append(idxs[i+1:].count(idxs[i]))
#     return idxs, rests, global_qidx


# def expand_question(dcur, global_qidx, pad_val=-1):
#     dextend, dlast = dict(), dict()
#     repeats = dcur["is_repeat"]
#     last = -1
#     dcur["qidxs"], dcur["rest"], global_qidx = add_qidx(dcur, global_qidx)
#     for i in range(len(repeats)):
#         if str(repeats[i]) == "0":
#             for key in dcur.keys():
#                 if key in ONE_KEYS:
#                     continue
#                 dlast[key] = dcur[key][0: i]
#         if i == 0:
#             for key in dcur.keys():
#                 if key in ONE_KEYS:
#                     continue
#                 dextend.setdefault(key, [])
#                 dextend[key].append([dcur[key][0]])
#             dextend.setdefault("selectmasks", [])
#             dextend["selectmasks"].append([pad_val])
#         else:
#             # print(f"i: {i}, dlast: {dlast.keys()}")
#             for key in dcur.keys():
#                 if key in ONE_KEYS:
#                     continue
#                 dextend.setdefault(key, [])
#                 if last == "0" and str(repeats[i]) == "0":
#                     dextend[key][-1] += [dcur[key][i]]
#                 else:
#                     dextend[key].append(dlast[key] + [dcur[key][i]])
#             dextend.setdefault("selectmasks", [])
#             if last == "0" and str(repeats[i]) == "0":
#                 dextend["selectmasks"][-1] += [1]
#             elif len(dlast["responses"]) == 0:  # the first question
#                 dextend["selectmasks"].append([pad_val])
#             else:
#                 dextend["selectmasks"].append(
#                     len(dlast["responses"]) * [pad_val] + [1])

#         last = str(repeats[i])

#     return dextend, global_qidx


# def generate_question_sequences(df, effective_keys, window=True, min_seq_len=3, maxlen=200, pad_val=-1):
#     if "questions" not in effective_keys or "concepts" not in effective_keys:
#         print(f"has no questions or concepts, has no question sequences!")
#         return False, None
#     save_keys = list(effective_keys) + \
#         ["selectmasks", "qidxs", "rest", "orirow"]
#     dres = {}  # "selectmasks": []}
#     global_qidx = -1
#     df["index"] = list(range(0, df.shape[0]))
#     for i, row in df.iterrows():
#         dcur = save_dcur(row, effective_keys)
#         dcur["orirow"] = [row["index"]] * len(dcur["responses"])

#         dexpand, global_qidx = expand_question(dcur, global_qidx)
#         seq_num = len(dexpand["responses"])
#         for j in range(seq_num):
#             curlen = len(dexpand["responses"][j])
#             if curlen < 2:  # 不预测第一个题
#                 continue
#             if curlen < maxlen:
#                 for key in dexpand:
#                     pad_dim = maxlen - curlen
# #                     print(key, j, len(dexpand[key]))
#                     paded_info = np.concatenate(
#                         [dexpand[key][j][0:], np.array([pad_val] * pad_dim)])
#                     dres.setdefault(key, [])
#                     dres[key].append(",".join([str(k) for k in paded_info]))
#                 for key in ONE_KEYS:
#                     dres.setdefault(key, [])
#                     dres[key].append(dcur[key])
#             else:
#                 # window
#                 if window:
#                     if dexpand["selectmasks"][j][maxlen-1] == 1:
#                         for key in dexpand:
#                             dres.setdefault(key, [])
#                             dres[key].append(
#                                 ",".join([str(k) for k in dexpand[key][j][0:maxlen]]))
#                         for key in ONE_KEYS:
#                             dres.setdefault(key, [])
#                             dres[key].append(dcur[key])

#                     for n in range(maxlen+1, curlen+1):
#                         if dexpand["selectmasks"][j][n-1] == 1:
#                             for key in dexpand:
#                                 dres.setdefault(key, [])
#                                 if key == "selectmasks":
#                                     dres[key].append(
#                                         ",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
#                                 else:
#                                     dres[key].append(
#                                         ",".join([str(k) for k in dexpand[key][j][n-maxlen: n]]))
#                             for key in ONE_KEYS:
#                                 dres.setdefault(key, [])
#                                 dres[key].append(dcur[key])
#                 else:
#                     # not window
#                     k = 0
#                     rest = curlen
#                     while curlen >= k + maxlen:
#                         rest = rest - maxlen
#                         if dexpand["selectmasks"][j][k + maxlen - 1] == 1:
#                             for key in dexpand:
#                                 dres.setdefault(key, [])
#                                 dres[key].append(
#                                     ",".join([str(s) for s in dexpand[key][j][k: k + maxlen]]))
#                             for key in ONE_KEYS:
#                                 dres.setdefault(key, [])
#                                 dres[key].append(dcur[key])
#                         k += maxlen
#                     if rest < min_seq_len:  # 剩下长度<min_seq_len不预测
#                         continue
#                     pad_dim = maxlen - rest
#                     for key in dexpand:
#                         dres.setdefault(key, [])
#                         paded_info = np.concatenate(
#                             [dexpand[key][j][k:], np.array([pad_val] * pad_dim)])
#                         dres[key].append(",".join([str(s)
#                                          for s in paded_info]))
#                     for key in ONE_KEYS:
#                         dres.setdefault(key, [])
#                         dres[key].append(dcur[key])
#                 #####

#     dfinal = dict()
#     for key in ALL_KEYS:
#         if key in save_keys:
#             # print(f"key: {key}, len: {len(dres[key])}")
#             dfinal[key] = dres[key]
#     finaldf = pd.DataFrame(dfinal)
#     return True, finaldf


# def save_id2idx(dkeyid2idx, save_path):
#     with open(save_path, "w+") as fout:
#         fout.write(json.dumps(dkeyid2idx))


# def write_config(dataset_name, dkeyid2idx, effective_keys, configf, dpath, k=5, min_seq_len=3, maxlen=200, flag=False, other_config={}):
#     input_type, num_q, num_c = [], 0, 0
#     if "questions" in effective_keys:
#         input_type.append("questions")
#         num_q = len(dkeyid2idx["questions"])
#     if "concepts" in effective_keys:
#         input_type.append("concepts")
#         num_c = len(dkeyid2idx["concepts"])
#     folds = list(range(0, k))
#     dconfig = {
#         "dpath": dpath,
#         "num_q": num_q,
#         "num_c": num_c,
#         "input_type": input_type,
#         "max_concepts": dkeyid2idx["max_concepts"],
#         "min_seq_len": min_seq_len,
#         "maxlen": maxlen,
#         "emb_path": "",
#         "train_valid_original_file": "train_valid.csv",
#         "train_valid_file": "train_valid_sequences.csv",
#         "folds": folds,
#         "test_original_file": "test.csv",
#         "test_file": "test_sequences.csv",
#         "test_window_file": "test_window_sequences.csv"
#     }
#     dconfig.update(other_config)
#     if flag:
#         dconfig["test_question_file"] = "test_question_sequences.csv"
#         dconfig["test_question_window_file"] = "test_question_window_sequences.csv"

#     # load old config
#     with open(configf) as fin:
#         read_text = fin.read()
#         if read_text.strip() == "":
#             data_config = {dataset_name: dconfig}
#         else:
#             data_config = json.loads(read_text)
#             if dataset_name in data_config:
#                 data_config[dataset_name].update(dconfig)
#             else:
#                 data_config[dataset_name] = dconfig

#     with open(configf, "w") as fout:
#         data = json.dumps(data_config, ensure_ascii=False, indent=4)
#         fout.write(data)


# def calStatistics(df, stares, key):
#     allin, allselect = 0, 0
#     allqs, allcs = set(), set()
#     for i, row in df.iterrows():
#         rs = row["responses"].split(",")
#         curlen = len(rs) - rs.count("-1")
#         allin += curlen
#         if "selectmasks" in row:
#             ss = row["selectmasks"].split(",")
#             slen = ss.count("1")
#             allselect += slen
#         if "concepts" in row:
#             cs = row["concepts"].split(",")
#             fc = list()
#             for c in cs:
#                 cc = c.split("_")
#                 fc.extend(cc)
#             curcs = set(fc) - {"-1"}
#             allcs |= curcs
#         if "questions" in row:
#             qs = row["questions"].split(",")
#             curqs = set(qs) - {"-1"}
#             allqs |= curqs
#     stares.append(",".join([str(s)
#                   for s in [key, allin, df.shape[0], allselect]]))
#     return allin, allselect, len(allqs), len(allcs), df.shape[0]


# def get_max_concepts(df):
#     max_concepts = 1
#     for i, row in df.iterrows():
#         cs = row["concepts"].split(",")
#         num_concepts = max([len(c.split("_")) for c in cs])
#         if num_concepts >= max_concepts:
#             max_concepts = num_concepts
#     return max_concepts


# def main(dname, fname, dataset_name, configf, min_seq_len=3, maxlen=200, kfold=5):
#     """split main function

#     Args:
#         dname (str): data folder path
#         fname (str): the data file used to split, needs 6 columns, format is: (NA indicates the dataset has no corresponding info)
#             uid,seqlen: 50121,4
#             quetion ids: NA
#             concept ids: 7014,7014,7014,7014
#             responses: 0,1,1,1
#             timestamps: NA
#             cost times: NA
#         dataset_name (str): dataset name
#         configf (str): the dataconfig file path
#         min_seq_len (int, optional): the min seqlen, sequences less than this value will be filtered out. Defaults to 3.
#         maxlen (int, optional): the max seqlen. Defaults to 200.
#         kfold (int, optional): the folds num needs to split. Defaults to 5.

#     """
#     stares = []

#     total_df, effective_keys = read_data(fname)
#     # cal max_concepts
#     if 'concepts' in effective_keys:
#         max_concepts = get_max_concepts(total_df)
#     else:
#         max_concepts = -1

#     oris, _, qs, cs, seqnum = calStatistics(total_df, stares, "original")
#     print("="*20)
#     print(
#         f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

#     total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)
#     total_df, dkeyid2idx = id_mapping(total_df)
#     dkeyid2idx["max_concepts"] = max_concepts

#     extends, _, qs, cs, seqnum = calStatistics(
#         total_df, stares, "extend multi")
#     print("="*20)
#     print(
#         f"after extend multi, total interactions: {extends}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

#     save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))
#     effective_keys.add("fold")
#     config = []
#     for key in ALL_KEYS:
#         if key in effective_keys:
#             config.append(key)
#     # train test split & generate sequences
#     train_df, test_df = train_test_split(total_df, 0.2)
#     splitdf = KFold_split(train_df, kfold)
#     # TODO
#     splitdf[config].to_csv(os.path.join(dname, "train_valid.csv"), index=None)
#     ins, ss, qs, cs, seqnum = calStatistics(
#         splitdf, stares, "original train+valid")
#     print(
#         f"train+valid original interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
#     split_seqs = generate_sequences(
#         splitdf, effective_keys, min_seq_len, maxlen)
#     ins, ss, qs, cs, seqnum = calStatistics(
#         split_seqs, stares, "train+valid sequences")
#     print(
#         f"train+valid sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
#     split_seqs.to_csv(os.path.join(
#         dname, "train_valid_sequences.csv"), index=None)
#     # print(f"split seqs dtypes: {split_seqs.dtypes}")

#     # add default fold -1 to test!
#     test_df["fold"] = [-1] * test_df.shape[0]
#     test_df['cidxs'] = get_inter_qidx(test_df)  # add index
#     test_seqs = generate_sequences(test_df, list(
#         effective_keys) + ['cidxs'], min_seq_len, maxlen)
#     ins, ss, qs, cs, seqnum = calStatistics(test_df, stares, "test original")
#     print(
#         f"original test interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
#     ins, ss, qs, cs, seqnum = calStatistics(
#         test_seqs, stares, "test sequences")
#     print(
#         f"test sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
#     print("="*20)

#     test_window_seqs = generate_window_sequences(
#         test_df, list(effective_keys) + ['cidxs'], maxlen)
#     flag, test_question_seqs = generate_question_sequences(
#         test_df, effective_keys, False, min_seq_len, maxlen)
#     flag, test_question_window_seqs = generate_question_sequences(
#         test_df, effective_keys, True, min_seq_len, maxlen)

#     test_df = test_df[config+['cidxs']]

#     test_df.to_csv(os.path.join(dname, "test.csv"), index=None)
#     test_seqs.to_csv(os.path.join(dname, "test_sequences.csv"), index=None)
#     test_window_seqs.to_csv(os.path.join(
#         dname, "test_window_sequences.csv"), index=None)

#     ins, ss, qs, cs, seqnum = calStatistics(
#         test_window_seqs, stares, "test window")
#     print(
#         f"test window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

#     if flag:
#         test_question_seqs.to_csv(os.path.join(
#             dname, "test_question_sequences.csv"), index=None)
#         test_question_window_seqs.to_csv(os.path.join(
#             dname, "test_question_window_sequences.csv"), index=None)

#         ins, ss, qs, cs, seqnum = calStatistics(
#             test_question_seqs, stares, "test question")
#         print(
#             f"test question interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
#         ins, ss, qs, cs, seqnum = calStatistics(
#             test_question_window_seqs, stares, "test question window")
#         print(
#             f"test question window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

#     write_config(dataset_name=dataset_name, dkeyid2idx=dkeyid2idx, effective_keys=effective_keys,
#                  configf=configf, dpath=dname, k=kfold, min_seq_len=min_seq_len, maxlen=maxlen, flag=flag)

#     print("="*20)
#     print("\n".join(stares))




import os
import sys
import pandas as pd
import numpy as np
import json
import copy

ALL_KEYS = ["fold", "uid", "questions", "concepts", "responses", "student_opts", "correct_opts", "timestamps",
            "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow", "cidxs"]
ONE_KEYS = ["fold", "uid"]


def read_data(fname, min_seq_len=3, response_set=[0, 1]):
    effective_keys = set()
    dres = dict()
    delstu, delnum, badr = 0, 0, 0
    goodnum = 0
    with open(fname, "r", encoding="utf8") as fin:
        i = 0
        lines = fin.readlines()
        dcur = dict()
        while i < len(lines):
            line = lines[i].strip()
            if i % 8 == 0:  # stuid  # 修改：从6改为8，因为增加了2个字段
                effective_keys.add("uid")
                tmps = line.split(",")
                if "(" in tmps[0]:
                    stuid, seq_len = tmps[0].replace('(', ''), int(tmps[2])
                else:
                    stuid, seq_len = tmps[0], int(tmps[1])
                if seq_len < min_seq_len:  # delete use seq len less than min_seq_len
                    i += 8  # 修改：从6改为8
                    dcur = dict()
                    delstu += 1
                    delnum += seq_len
                    continue
                dcur["uid"] = stuid
                goodnum += seq_len
            elif i % 8 == 1:  # question ids / names  # 修改：从6改为8
                qs = []
                # if line.find("NA") == -1:
                if "NA" not in line.split(","):
                    effective_keys.add("questions")
                    qs = line.split(",")
                    
                dcur["questions"] = qs
            elif i % 8 == 2:  # concept ids / names  # 修改：从6改为8
                cs = []
                if line.find("NA") == -1:
                    effective_keys.add("concepts")
                    cs = line.split(",")
                dcur["concepts"] = cs
            elif i % 8 == 3:  # responses  # 修改：从6改为8
                effective_keys.add("responses")
                rs = []
                if line.find("NA") == -1:
                    flag = True
                    for r in line.split(","):
                        try:
                            r = int(r)
                            if r not in response_set:  # check if r in response set.
                                print(f"error response in line: {i}")
                                flag = False
                                break
                            rs.append(r)
                        except:
                            print(f"error response in line: {i}")
                            flag = False
                            break
                    if not flag:
                        i += 5  # 修改：跳过剩余5行（原来是3行）
                        dcur = dict()
                        badr += 1
                        continue
                dcur["responses"] = rs
            elif i % 8 == 4:  # student_opts  # 新增：处理answer字段
                effective_keys.add("student_opts")
                ans = []
                if line.find("NA") == -1:
                    ans = line.split(",")
                dcur["student_opts"] = ans
            elif i % 8 == 5:  # correct_opts  # 新增：处理correct_answer字段
                effective_keys.add("correct_opts")
                cans = []
                if line.find("NA") == -1:
                    cans = line.split(",")
                dcur["correct_opts"] = cans
            elif i % 8 == 6:  # timestamps  # 修改：从6改为8
                ts = []
                if line.find("NA") == -1:
                    effective_keys.add("timestamps")
                    ts = line.split(",")
                dcur["timestamps"] = ts
            elif i % 8 == 7:  # usets  # 修改：从6改为8
                usets = []
                if line.find("NA") == -1:
                    effective_keys.add("usetimes")
                    usets = line.split(",")
                dcur["usetimes"] = usets

                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key != "uid":
                        dres[key].append(",".join([str(k) for k in dcur[key]]))
                    else:
                        dres[key].append(dcur[key])
                dcur = dict()
            i += 1
    df = pd.DataFrame(dres)
    print(
        f"delete bad stu num of len: {delstu}, delete interactions: {delnum}, of r: {badr}, good num: {goodnum}")
    return df, effective_keys


def extend_multi_concepts(df, effective_keys):
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print("has no questions or concepts! return original.")
        return df, effective_keys
    extend_keys = set(df.columns) - {"uid"}

    dres = {"uid": df["uid"]}
    for _, row in df.iterrows():
        dextend_infos = dict()
        for key in extend_keys:
            dextend_infos[key] = row[key].split(",")
        dextend_res = dict()
        for i in range(len(dextend_infos["questions"])):
            dextend_res.setdefault("is_repeat", [])
            if dextend_infos["concepts"][i].find("_") != -1:
                ids = dextend_infos["concepts"][i].split("_")
                dextend_res.setdefault("concepts", [])
                dextend_res["concepts"].extend(ids)
                for key in extend_keys:
                    if key != "concepts":
                        dextend_res.setdefault(key, [])
                        dextend_res[key].extend(
                            [dextend_infos[key][i]] * len(ids))
                dextend_res["is_repeat"].extend(
                    ["0"] + ["1"] * (len(ids) - 1))  # 1: repeat, 0: original
            else:
                for key in extend_keys:
                    dextend_res.setdefault(key, [])
                    dextend_res[key].append(dextend_infos[key][i])
                dextend_res["is_repeat"].append("0")
        for key in dextend_res:
            dres.setdefault(key, [])
            dres[key].append(",".join(dextend_res[key]))

    finaldf = pd.DataFrame(dres)
    effective_keys.add("is_repeat")
    return finaldf, effective_keys




# def extend_multi_concepts(df, effective_keys):
#     if "questions" not in effective_keys or "concepts" not in effective_keys:
#         print("has no questions or concepts! return original.")
#         return df, effective_keys
#     extend_keys = set(df.columns) - {"uid"}

#     dres = {"uid": df["uid"]}
#     for idx, row in df.iterrows():
#         try:
#             dextend_infos = dict()
#             for key in extend_keys:
#                 dextend_infos[key] = row[key].split(",")
            
#             # 检查字段长度一致性
#             lengths = {key: len(dextend_infos[key]) for key in extend_keys}
#             questions_len = lengths.get("questions", 0)
            
#             # 检查是否有长度不一致的字段
#             inconsistent_keys = []
#             for key, length in lengths.items():
#                 if length != questions_len:
#                     inconsistent_keys.append(f"{key}({length})")
            
#             if inconsistent_keys:
#                 print(f"\n❌ 数据不一致 - 行 {idx}:")
#                 print(f"  uid: {row['uid']}")
#                 print(f"  questions长度: {questions_len}")
#                 print(f"  不一致字段: {', '.join(inconsistent_keys)}")
#                 print(f"  详细长度: {lengths}")
                
#                 # 显示原始数据（截断显示）
#                 print("  原始数据预览:")
#                 for key in extend_keys:
#                     content = str(row[key])[:100] + "..." if len(str(row[key])) > 100 else str(row[key])
#                     print(f"    {key}: {content}")
                
#                 # 使用最小长度继续处理
#                 safe_length = min(lengths.values())
#                 print(f"  使用安全长度: {safe_length}")
#             else:
#                 safe_length = questions_len
            
#             dextend_res = dict()
#             for i in range(safe_length):
#                 try:
#                     dextend_res.setdefault("is_repeat", [])
                    
#                     # 在访问数组元素前添加边界检查
#                     concept_value = dextend_infos["concepts"][i] if i < len(dextend_infos["concepts"]) else "NA"
                    
#                     if concept_value.find("_") != -1:
#                         ids = concept_value.split("_")
#                         dextend_res.setdefault("concepts", [])
#                         dextend_res["concepts"].extend(ids)
#                         for key in extend_keys:
#                             if key != "concepts":
#                                 dextend_res.setdefault(key, [])
#                                 # 添加边界检查
#                                 value = dextend_infos[key][i] if i < len(dextend_infos[key]) else "NA"
#                                 dextend_res[key].extend([value] * len(ids))
#                         dextend_res["is_repeat"].extend(
#                             ["0"] + ["1"] * (len(ids) - 1))
#                     else:
#                         for key in extend_keys:
#                             dextend_res.setdefault(key, [])
#                             # 添加边界检查 - 这里是出错的地方！
#                             if i < len(dextend_infos[key]):
#                                 dextend_res[key].append(dextend_infos[key][i])
#                             else:
#                                 print(f"\n🚨 索引越界警告 - 行 {idx}, 位置 {i}, 字段 {key}:")
#                                 print(f"  尝试访问索引 {i}，但 {key} 长度只有 {len(dextend_infos[key])}")
#                                 print(f"  uid: {row['uid']}")
#                                 print(f"  字段内容: {dextend_infos[key]}")
#                                 dextend_res[key].append("NA")  # 用NA填充
#                         dextend_res["is_repeat"].append("0")
                        
#                 except Exception as inner_e:
#                     print(f"\n🚨 内层循环错误 - 行 {idx}, 位置 {i}:")
#                     print(f"  错误: {inner_e}")
#                     print(f"  uid: {row['uid']}")
#                     print(f"  字段长度: {lengths}")
#                     raise inner_e
            
#             for key in dextend_res:
#                 dres.setdefault(key, [])
#                 dres[key].append(",".join(dextend_res[key]))
                
#         except Exception as e:
#             print(f"\n💥 外层循环严重错误 - 行 {idx}:")
#             print(f"  错误类型: {type(e).__name__}")
#             print(f"  错误信息: {e}")
#             print(f"  uid: {row.get('uid', 'UNKNOWN')}")
#             print("  这一行将被跳过...")
            
#             # 跳过这一行，继续处理下一行
#             continue

#     finaldf = pd.DataFrame(dres)
#     effective_keys.add("is_repeat")
#     return finaldf, effective_keys


def id_mapping(df):
    id_keys = ["questions", "concepts", "uid"]
    dres = dict()
    dkeyid2idx = dict()
    print(f"df.columns: {df.columns}")
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]
    for i, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            dkeyid2idx.setdefault(key, dict())
            dres.setdefault(key, [])
            curids = []
            for id in row[key].split(","):
                if id not in dkeyid2idx[key]:
                    dkeyid2idx[key][id] = len(dkeyid2idx[key])
                curids.append(str(dkeyid2idx[key][id]))
            dres[key].append(",".join(curids))
    finaldf = pd.DataFrame(dres)
    return finaldf, dkeyid2idx


def train_test_split(df, test_ratio=0.2):
    df = df.sample(frac=1.0, random_state=1024)
    datanum = df.shape[0]
    test_num = int(datanum * test_ratio)
    train_num = datanum - test_num
    train_df = df[0:train_num]
    test_df = df[train_num:]
    # report
    print(
        f"total num: {datanum}, train+valid num: {train_num}, test num: {test_num}")
    return train_df, test_df


def KFold_split(df, k=5):
    df = df.sample(frac=1.0, random_state=1024)
    datanum = df.shape[0]
    test_ratio = 1 / k
    test_num = int(datanum * test_ratio)
    rest = datanum % k

    start = 0
    folds = []
    for i in range(0, k):
        if rest > 0:
            end = start + test_num + 1
            rest -= 1
        else:
            end = start + test_num
        folds.extend([i] * (end - start))
        print(f"fold: {i+1}, start: {start}, end: {end}, total num: {datanum}")
        start = end
    # report
    finaldf = copy.deepcopy(df)
    finaldf["fold"] = folds
    return finaldf


def save_dcur(row, effective_keys):
    dcur = dict()
    for key in effective_keys:
        if key not in ONE_KEYS:
            # [int(i) for i in row[key].split(",")]
            dcur[key] = row[key].split(",")
        else:
            dcur[key] = row[key]
    return dcur


def generate_sequences(df, effective_keys, min_seq_len=3, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    dropnum = 0
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)

        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        while lenrs >= j + maxlen:
            rest = rest - (maxlen)
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    # [str(k) for k in dcur[key][j: j + maxlen]]))
                    dres[key].append(",".join(dcur[key][j: j + maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))

            j += maxlen
        if rest < min_seq_len:  # delete sequence len less than min_seq_len
            dropnum += rest
            continue

        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key not in ONE_KEYS:
                paded_info = np.concatenate(
                    [dcur[key][j:], np.array([pad_val] * pad_dim)])
                dres[key].append(",".join([str(k) for k in paded_info]))
            else:
                dres[key].append(dcur[key])
        dres["selectmasks"].append(
            ",".join(["1"] * rest + [str(pad_val)] * pad_dim))

    # after preprocess data, report
    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    print(f"dropnum: {dropnum}")
    return finaldf


def generate_window_sequences(df, effective_keys, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        lenrs = len(dcur["responses"])
        if lenrs > maxlen:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    # [str(k) for k in dcur[key][0: maxlen]]))
                    dres[key].append(",".join(dcur[key][0: maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            for j in range(maxlen+1, lenrs+1):
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key not in ONE_KEYS:
                        dres[key].append(",".join([str(k)
                                         for k in dcur[key][j-maxlen: j]]))
                    else:
                        dres[key].append(dcur[key])
                dres["selectmasks"].append(
                    ",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
        else:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    pad_dim = maxlen - lenrs
                    paded_info = np.concatenate(
                        [dcur[key][0:], np.array([pad_val] * pad_dim)])
                    dres[key].append(",".join([str(k) for k in paded_info]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(
                ",".join(["1"] * lenrs + [str(pad_val)] * pad_dim))

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            # print(f"key: {key}, len: {len(dres[key])}")
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return finaldf


def get_inter_qidx(df):
    """add global id for each interaction"""
    qidx_ids = []
    bias = 0
    inter_num = 0
    for _, row in df.iterrows():
        ids_list = [str(x+bias)
                    for x in range(len(row['responses'].split(',')))]
        inter_num += len(ids_list)
        ids = ",".join(ids_list)
        qidx_ids.append(ids)
        bias += len(ids_list)
    assert inter_num-1 == int(ids_list[-1])

    return qidx_ids


def add_qidx(dcur, global_qidx):
    idxs, rests = [], []
    # idx = -1
    for r in dcur["is_repeat"]:
        if str(r) == "0":
            global_qidx += 1
        idxs.append(global_qidx)
    # print(dcur["is_repeat"])
    # print(f"idxs: {idxs}")
    # print("="*20)
    for i in range(0, len(idxs)):
        rests.append(idxs[i+1:].count(idxs[i]))
    return idxs, rests, global_qidx


def expand_question(dcur, global_qidx, pad_val=-1):
    dextend, dlast = dict(), dict()
    repeats = dcur["is_repeat"]
    last = -1
    dcur["qidxs"], dcur["rest"], global_qidx = add_qidx(dcur, global_qidx)
    for i in range(len(repeats)):
        if str(repeats[i]) == "0":
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dlast[key] = dcur[key][0: i]
        if i == 0:
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dextend.setdefault(key, [])
                dextend[key].append([dcur[key][0]])
            dextend.setdefault("selectmasks", [])
            dextend["selectmasks"].append([pad_val])
        else:
            # print(f"i: {i}, dlast: {dlast.keys()}")
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dextend.setdefault(key, [])
                if last == "0" and str(repeats[i]) == "0":
                    dextend[key][-1] += [dcur[key][i]]
                else:
                    dextend[key].append(dlast[key] + [dcur[key][i]])
            dextend.setdefault("selectmasks", [])
            if last == "0" and str(repeats[i]) == "0":
                dextend["selectmasks"][-1] += [1]
            elif len(dlast["responses"]) == 0:  # the first question
                dextend["selectmasks"].append([pad_val])
            else:
                dextend["selectmasks"].append(
                    len(dlast["responses"]) * [pad_val] + [1])

        last = str(repeats[i])

    return dextend, global_qidx


def generate_question_sequences(df, effective_keys, window=True, min_seq_len=3, maxlen=200, pad_val=-1):
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print(f"has no questions or concepts, has no question sequences!")
        return False, None
    save_keys = list(effective_keys) + \
        ["selectmasks", "qidxs", "rest", "orirow"]
    dres = {}  # "selectmasks": []}
    global_qidx = -1
    df["index"] = list(range(0, df.shape[0]))
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        dcur["orirow"] = [row["index"]] * len(dcur["responses"])

        dexpand, global_qidx = expand_question(dcur, global_qidx)
        seq_num = len(dexpand["responses"])
        for j in range(seq_num):
            curlen = len(dexpand["responses"][j])
            if curlen < 2:  # 不预测第一个题
                continue
            if curlen < maxlen:
                for key in dexpand:
                    pad_dim = maxlen - curlen
#                     print(key, j, len(dexpand[key]))
                    paded_info = np.concatenate(
                        [dexpand[key][j][0:], np.array([pad_val] * pad_dim)])
                    dres.setdefault(key, [])
                    dres[key].append(",".join([str(k) for k in paded_info]))
                for key in ONE_KEYS:
                    dres.setdefault(key, [])
                    dres[key].append(dcur[key])
            else:
                # window
                if window:
                    if dexpand["selectmasks"][j][maxlen-1] == 1:
                        for key in dexpand:
                            dres.setdefault(key, [])
                            dres[key].append(
                                ",".join([str(k) for k in dexpand[key][j][0:maxlen]]))
                        for key in ONE_KEYS:
                            dres.setdefault(key, [])
                            dres[key].append(dcur[key])

                    for n in range(maxlen+1, curlen+1):
                        if dexpand["selectmasks"][j][n-1] == 1:
                            for key in dexpand:
                                dres.setdefault(key, [])
                                if key == "selectmasks":
                                    dres[key].append(
                                        ",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
                                else:
                                    dres[key].append(
                                        ",".join([str(k) for k in dexpand[key][j][n-maxlen: n]]))
                            for key in ONE_KEYS:
                                dres.setdefault(key, [])
                                dres[key].append(dcur[key])
                else:
                    # not window
                    k = 0
                    rest = curlen
                    while curlen >= k + maxlen:
                        rest = rest - maxlen
                        if dexpand["selectmasks"][j][k + maxlen - 1] == 1:
                            for key in dexpand:
                                dres.setdefault(key, [])
                                dres[key].append(
                                    ",".join([str(s) for s in dexpand[key][j][k: k + maxlen]]))
                            for key in ONE_KEYS:
                                dres.setdefault(key, [])
                                dres[key].append(dcur[key])
                        k += maxlen
                    if rest < min_seq_len:  # 剩下长度<min_seq_len不预测
                        continue
                    pad_dim = maxlen - rest
                    for key in dexpand:
                        dres.setdefault(key, [])
                        paded_info = np.concatenate(
                            [dexpand[key][j][k:], np.array([pad_val] * pad_dim)])
                        dres[key].append(",".join([str(s)
                                         for s in paded_info]))
                    for key in ONE_KEYS:
                        dres.setdefault(key, [])
                        dres[key].append(dcur[key])
                #####

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            # print(f"key: {key}, len: {len(dres[key])}")
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return True, finaldf


def save_id2idx(dkeyid2idx, save_path):
    with open(save_path, "w+") as fout:
        fout.write(json.dumps(dkeyid2idx))


def write_config(dataset_name, dkeyid2idx, effective_keys, configf, dpath, k=5, min_seq_len=3, maxlen=200, flag=False, other_config={}):
    input_type, num_q, num_c = [], 0, 0
    if "questions" in effective_keys:
        input_type.append("questions")
        num_q = len(dkeyid2idx["questions"])
    if "concepts" in effective_keys:
        input_type.append("concepts")
        num_c = len(dkeyid2idx["concepts"])
    folds = list(range(0, k))
    dconfig = {
        "dpath": dpath,
        "num_q": num_q,
        "num_c": num_c,
        "input_type": input_type,
        "max_concepts": dkeyid2idx["max_concepts"],
        "min_seq_len": min_seq_len,
        "maxlen": maxlen,
        "emb_path": "",
        "train_valid_original_file": "train_valid.csv",
        "train_valid_file": "train_valid_sequences.csv",
        "folds": folds,
        "test_original_file": "test.csv",
        "test_file": "test_sequences.csv",
        "test_window_file": "test_window_sequences.csv"
    }
    dconfig.update(other_config)
    if flag:
        dconfig["test_question_file"] = "test_question_sequences.csv"
        dconfig["test_question_window_file"] = "test_question_window_sequences.csv"

    # load old config
    with open(configf) as fin:
        read_text = fin.read()
        if read_text.strip() == "":
            data_config = {dataset_name: dconfig}
        else:
            data_config = json.loads(read_text)
            if dataset_name in data_config:
                data_config[dataset_name].update(dconfig)
            else:
                data_config[dataset_name] = dconfig

    with open(configf, "w") as fout:
        data = json.dumps(data_config, ensure_ascii=False, indent=4)
        fout.write(data)


def calStatistics(df, stares, key):
    allin, allselect = 0, 0
    allqs, allcs = set(), set()
    for i, row in df.iterrows():
        rs = row["responses"].split(",")
        curlen = len(rs) - rs.count("-1")
        allin += curlen
        if "selectmasks" in row:
            ss = row["selectmasks"].split(",")
            slen = ss.count("1")
            allselect += slen
        if "concepts" in row:
            cs = row["concepts"].split(",")
            fc = list()
            for c in cs:
                cc = c.split("_")
                fc.extend(cc)
            curcs = set(fc) - {"-1"}
            allcs |= curcs
        if "questions" in row:
            qs = row["questions"].split(",")
            curqs = set(qs) - {"-1"}
            allqs |= curqs
    stares.append(",".join([str(s)
                  for s in [key, allin, df.shape[0], allselect]]))
    return allin, allselect, len(allqs), len(allcs), df.shape[0]


def get_max_concepts(df):
    max_concepts = 1
    for i, row in df.iterrows():
        cs = row["concepts"].split(",")
        num_concepts = max([len(c.split("_")) for c in cs])
        if num_concepts >= max_concepts:
            max_concepts = num_concepts
    return max_concepts


def main(dname, fname, dataset_name, configf, min_seq_len=3, maxlen=200, kfold=5):
    """split main function

    Args:
        dname (str): data folder path
        fname (str): the data file used to split, needs 8 columns, format is: (NA indicates the dataset has no corresponding info)  # 修改：从6改为8
            uid,seqlen: 50121,4
            quetion ids: NA
            concept ids: 7014,7014,7014,7014
            responses: 0,1,1,1
            student_opts: 0,1,2,3  # 新增
            correct_opts: 1,1,2,3  # 新增
            timestamps: NA
            cost times: NA
        dataset_name (str): dataset name
        configf (str): the dataconfig file path
        min_seq_len (int, optional): the min seqlen, sequences less than this value will be filtered out. Defaults to 3.
        maxlen (int, optional): the max seqlen. Defaults to 200.
        kfold (int, optional): the folds num needs to split. Defaults to 5.

    """
    stares = []

    total_df, effective_keys = read_data(fname)
    # cal max_concepts
    if 'concepts' in effective_keys:
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1

    oris, _, qs, cs, seqnum = calStatistics(total_df, stares, "original")
    print("="*20)
    print(
        f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)
    total_df, dkeyid2idx = id_mapping(total_df)
    dkeyid2idx["max_concepts"] = max_concepts

    extends, _, qs, cs, seqnum = calStatistics(
        total_df, stares, "extend multi")
    print("="*20)
    print(
        f"after extend multi, total interactions: {extends}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))
    effective_keys.add("fold")
    config = []
    for key in ALL_KEYS:
        if key in effective_keys:
            config.append(key)
    # train test split & generate sequences
    train_df, test_df = train_test_split(total_df, 0.2)
    splitdf = KFold_split(train_df, kfold)
    # TODO
    splitdf[config].to_csv(os.path.join(dname, "train_valid.csv"), index=None)
    ins, ss, qs, cs, seqnum = calStatistics(
        splitdf, stares, "original train+valid")
    print(
        f"train+valid original interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    split_seqs = generate_sequences(
        splitdf, effective_keys, min_seq_len, maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(
        split_seqs, stares, "train+valid sequences")
    print(
        f"train+valid sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    split_seqs.to_csv(os.path.join(
        dname, "train_valid_sequences.csv"), index=None)
    # print(f"split seqs dtypes: {split_seqs.dtypes}")

    # add default fold -1 to test!
    test_df["fold"] = [-1] * test_df.shape[0]
    test_df['cidxs'] = get_inter_qidx(test_df)  # add index
    test_seqs = generate_sequences(test_df, list(
        effective_keys) + ['cidxs'], min_seq_len, maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(test_df, stares, "test original")
    print(
        f"original test interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    ins, ss, qs, cs, seqnum = calStatistics(
        test_seqs, stares, "test sequences")
    print(
        f"test sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    print("="*20)

    test_window_seqs = generate_window_sequences(
        test_df, list(effective_keys) + ['cidxs'], maxlen)
    flag, test_question_seqs = generate_question_sequences(
        test_df, effective_keys, False, min_seq_len, maxlen)
    flag, test_question_window_seqs = generate_question_sequences(
        test_df, effective_keys, True, min_seq_len, maxlen)

    test_df = test_df[config+['cidxs']]

    test_df.to_csv(os.path.join(dname, "test.csv"), index=None)
    test_seqs.to_csv(os.path.join(dname, "test_sequences.csv"), index=None)
    test_window_seqs.to_csv(os.path.join(
        dname, "test_window_sequences.csv"), index=None)

    ins, ss, qs, cs, seqnum = calStatistics(
        test_window_seqs, stares, "test window")
    print(
        f"test window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    if flag:
        test_question_seqs.to_csv(os.path.join(
            dname, "test_question_sequences.csv"), index=None)
        test_question_window_seqs.to_csv(os.path.join(
            dname, "test_question_window_sequences.csv"), index=None)

        ins, ss, qs, cs, seqnum = calStatistics(
            test_question_seqs, stares, "test question")
        print(
            f"test question interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
        ins, ss, qs, cs, seqnum = calStatistics(
            test_question_window_seqs, stares, "test question window")
        print(
            f"test question window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    write_config(dataset_name=dataset_name, dkeyid2idx=dkeyid2idx, effective_keys=effective_keys,
                 configf=configf, dpath=dname, k=kfold, min_seq_len=min_seq_len, maxlen=maxlen, flag=flag)

    print("="*20)
    print("\n".join(stares))