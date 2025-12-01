# import pandas as pd
# import numpy as np
# from .utils import sta_infos, write_txt, format_list2str, change2timestamp, replace_text

# KEYS = ["user_id", "kc_en", "question_id"]

# def process_knowledge_list(kc):
#     """
#     Handling knowledge list strings, unifying formats and replacing commas, underscores
#     """
#     return kc.replace(', ', '@@').replace(',', '@')

# def read_data_from_csv(read_file, write_file):
#     """处理XXJ数据集"""
#     stares = []
    
#     # 读取数据
#     df = pd.read_csv(read_file, encoding='utf-8')
    
#     # 处理用户ID中的下划线
#     df.loc[:, 'user_id'] = df['user_id'].apply(replace_text)  # 将下划线替换为'####'
#     df.loc[:, 'question_id'] = df['question_id'].astype(str).apply(replace_text)
#     # 原始数据统计
#     ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
#     print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
#     # 数据清洗
#     df['tmp_index'] = range(len(df))
#     _df = df.copy()
#     # _df = df[df['答题状态'] != '半对'].copy()
    
#     # 处理知识点列表
#     # _df.loc[:, 'knowledge_list'] = _df['knowledge_list'].apply(process_knowledge_list)
#     _df.loc[:, 'kc_en'] = _df['kc_en'].apply(process_knowledge_list)
#     unique_kc = _df['kc_en'].unique().tolist()
#     print(len(unique_kc), "unique knowledge points found")
#     # # 把上述列表中所有的元素变成小写
#     # for i in range(len(unique_kc)):
#     #     unique_kc[i] = unique_kc[i].lower()
#     # print(len(unique_kc), "unique knowledge points found after lower")
#     # import sys
#     # sys.exit(0)
    
#     # _df = _df[_df['knowledge_list'] != ""]
#     # _df = _df.dropna(subset=['user_id', 'question_id', 'knowledge_list', 'is_correct', 'created_at'])
#     _df = _df.dropna(subset=['user_id', 'question_id', 'kc_en', 'is_correct', 'created_at'])
    
#     # 验证数据格式
#     assert all(_df['is_correct'].isin([0, 1])), "答案格式不正确"
    
#     # 转换时间戳
#     _df.loc[:, 'created_at'] = _df.loc[:, 'created_at'].apply(lambda t: change2timestamp(t, False))
    
#     # 清洗后的数据统计
#     ins, us, qs, cs, avgins, avgcq, na = sta_infos(_df, KEYS, stares)
#     print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
#     # 按用户分组处理数据
#     user_inters = []
#     ui_df = _df.groupby('user_id', sort=False)
    
#     for ui in ui_df:
#         user, tmp_inter = ui[0], ui[1]
#         tmp_inter = tmp_inter.sort_values(by=['created_at', 'tmp_index'])
        
#         seq_len = len(tmp_inter)
#         seq_problems = tmp_inter['question_id'].astype(str).tolist()
#         # seq_skills = tmp_inter['knowledge_list'].tolist()  # 已经是下划线连接的字符串
#         seq_skills = tmp_inter['kc_en'].tolist()
#         seq_ans = tmp_inter['is_correct'].astype(str).tolist()
#         seq_start_time = tmp_inter['created_at'].astype(str).tolist()
#         seq_response_cost = ['NA']
        
#         assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_start_time)
        
#         user_inters.append([
#             [str(user), str(seq_len)],
#             format_list2str(seq_problems),
#             format_list2str(seq_skills),
#             format_list2str(seq_ans),
#             seq_start_time,
#             seq_response_cost
#         ])
    
#     write_txt(write_file, user_inters)
#     print("\n".join(stares))
    
#     return


import pandas as pd
import numpy as np
from .utils import sta_infos, write_txt, format_list2str, change2timestamp, replace_text

KEYS = ["user_id", "kc_en", "question_id"]

def process_knowledge_list(kc):
    """
    Handling knowledge list strings, unifying formats and replacing commas, underscores
    """
    return kc.replace(', ', '@@').replace(',', '@')

def letter_to_number(letter):
    """
    将字母答案转换为数字：A->0, B->1, C->2, D->3, E->4, ...
    """
    if pd.isna(letter) or letter == '' or letter is None:
        return 'NA'
    letter = str(letter).strip().upper()
    if letter in 'ABCDEFGHIJ':
        return str(ord(letter) - ord('A'))
    return 'NA'

def read_data_from_csv(read_file, write_file):
    """处理XXJ数据集"""
    stares = []
    
    # 读取数据
    df = pd.read_csv(read_file, encoding='utf-8')
    
    # 处理用户ID中的下划线
    df.loc[:, 'user_id'] = df['user_id'].apply(replace_text)  # 将下划线替换为'####'
    df.loc[:, 'question_id'] = df['question_id'].astype(str).apply(replace_text)
    # 原始数据统计
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    # 数据清洗
    df['tmp_index'] = range(len(df))
    _df = df.copy()
    # _df = df[df['答题状态'] != '半对'].copy()
    
    # 处理知识点列表
    # _df.loc[:, 'knowledge_list'] = _df['knowledge_list'].apply(process_knowledge_list)
    _df.loc[:, 'kc_en'] = _df['kc_en'].apply(process_knowledge_list)
    unique_kc = _df['kc_en'].unique().tolist()
    print(len(unique_kc), "unique knowledge points found")
    # # 把上述列表中所有的元素变成小写
    # for i in range(len(unique_kc)):
    #     unique_kc[i] = unique_kc[i].lower()
    # print(len(unique_kc), "unique knowledge points found after lower")
    # import sys
    # sys.exit(0)
    
    # _df = _df[_df['knowledge_list'] != ""]
    # _df = _df.dropna(subset=['user_id', 'question_id', 'knowledge_list', 'is_correct', 'created_at'])
    _df = _df.dropna(subset=['user_id', 'question_id', 'kc_en', 'is_correct', 'created_at'])
    
    # 验证数据格式
    assert all(_df['is_correct'].isin([0, 1])), "答案格式不正确"
    
    # 转换时间戳
    _df.loc[:, 'created_at'] = _df.loc[:, 'created_at'].apply(lambda t: change2timestamp(t, False))
    
    # 清洗后的数据统计
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(_df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    # 按用户分组处理数据
    user_inters = []
    ui_df = _df.groupby('user_id', sort=False)
    
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=['created_at', 'tmp_index'])
        
        seq_len = len(tmp_inter)
        seq_problems = tmp_inter['question_id'].astype(str).tolist()
        # seq_skills = tmp_inter['knowledge_list'].tolist()  # 已经是下划线连接的字符串
        seq_skills = tmp_inter['kc_en'].tolist()
        seq_ans = tmp_inter['is_correct'].astype(str).tolist()
        seq_start_time = tmp_inter['created_at'].astype(str).tolist()
        seq_response_cost = ['NA']
        
        # 添加answer和correct_answer的处理
        seq_answer = tmp_inter['answer'].apply(letter_to_number).tolist()
        seq_correct_answer = tmp_inter['correct_answer'].apply(letter_to_number).tolist()
        
        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_start_time)
        assert seq_len == len(seq_answer) == len(seq_correct_answer)
        
        user_inters.append([
            [str(user), str(seq_len)],
            format_list2str(seq_problems),
            format_list2str(seq_skills),
            format_list2str(seq_ans),
            format_list2str(seq_answer),
            format_list2str(seq_correct_answer),
            seq_start_time,
            seq_response_cost
        ])
    
    write_txt(write_file, user_inters)
    print("\n".join(stares))
    
    return