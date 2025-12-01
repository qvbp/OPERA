# import pandas as pd
# import numpy as np
# from .utils import sta_infos, write_txt, format_list2str, change2timestamp, replace_text

# KEYS = ["user_id", "knowledge_list", "question_id"]

# def process_knowledge_list(knowledge_str):
#     """
#     处理知识点列表字符串，统一格式并替换逗号，下划线
#     """
#     try:
#         knowledge_list = eval(knowledge_str)
#         if not knowledge_list:
#             return ""
        
#         # 处理每个知识点
#         processed_list = []
#         for kc in knowledge_list:
#             # 替换英文逗号和中文逗号为破折号
#             processed_kc = kc.replace(',', '@@@@').replace('，', '@@@@')
#             processed_list.append(processed_kc)
            
#         # 用下划线连接所有处理后的知识点
#         return "_".join(processed_list)
#     except Exception as e:
#         print(f"Error processing knowledge string: {knowledge_str}, Error: {e}")
#         return ""

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
#     _df.loc[:, 'knowledge_list'] = _df['knowledge_list'].apply(process_knowledge_list)
    
#     _df = _df[_df['knowledge_list'] != ""]
#     _df = _df.dropna(subset=['user_id', 'question_id', 'knowledge_list', 'is_correct', 'created_at'])
    
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
#         seq_skills = tmp_inter['knowledge_list'].tolist()  # 已经是下划线连接的字符串
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

KEYS = ["user_id", "knowledge_list", "question_id"]

def answer_to_number(answer):
    """
    将选择题答案A/B/C/D/E转换为数字0/1/2/3/4
    """
    if pd.isna(answer) or answer == '':
        return 'NA'
    
    answer = str(answer).strip().upper()
    if answer in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        return str(ord(answer) - ord('A'))
    else:
        return 'NA'

def process_knowledge_list(knowledge_str):
    """
    处理知识点列表字符串，统一格式并替换逗号，下划线
    """
    try:
        knowledge_list = eval(knowledge_str)
        if not knowledge_list:
            return ""
        
        # 处理每个知识点
        processed_list = []
        for kc in knowledge_list:
            # 替换英文逗号和中文逗号为破折号
            processed_kc = kc.replace(',', '@@@@').replace('，', '@@@@')
            processed_list.append(processed_kc)
            
        # 用下划线连接所有处理后的知识点
        return "_".join(processed_list)
    except Exception as e:
        print(f"Error processing knowledge string: {knowledge_str}, Error: {e}")
        return ""

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
    _df.loc[:, 'knowledge_list'] = _df['knowledge_list'].apply(process_knowledge_list)
    
    _df = _df[_df['knowledge_list'] != ""]
    _df = _df.dropna(subset=['user_id', 'question_id', 'knowledge_list', 'is_correct', 'created_at'])
    
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
        seq_skills = tmp_inter['knowledge_list'].tolist()  # 已经是下划线连接的字符串
        seq_ans = tmp_inter['is_correct'].astype(str).tolist()
        seq_start_time = tmp_inter['created_at'].astype(str).tolist()
        seq_response_cost = ['NA']
        
        # 新增：处理answer和correct_answer字段
        seq_answer = [answer_to_number(ans) for ans in tmp_inter['answer'].tolist()]
        seq_correct_answer = [answer_to_number(ans) for ans in tmp_inter['correct_answer'].tolist()]
        
        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_start_time)
        assert seq_len == len(seq_answer) == len(seq_correct_answer)
        
        user_inters.append([
            [str(user), str(seq_len)],
            format_list2str(seq_problems),
            format_list2str(seq_skills),
            format_list2str(seq_ans),
            format_list2str(seq_answer),        # 新增：学生答案序列
            format_list2str(seq_correct_answer), # 新增：正确答案序列
            seq_start_time,
            seq_response_cost
        ])
    
    write_txt(write_file, user_inters)
    print("\n".join(stares))
    
    return