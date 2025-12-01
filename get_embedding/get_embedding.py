import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import warnings
import argparse
import sys
import os
warnings.filterwarnings("ignore")

class QuestionEmbedder:
    def __init__(self, model_path, max_length=512):
        """
        初始化BERT模型和tokenizer
        
        Args:
            model_path: 本地BERT模型路径
            max_length: 最大序列长度
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer和model
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.max_length = max_length
        
    def get_text_embedding(self, text):
        """
        获取单个文本的BERT embedding
        
        Args:
            text: 输入文本
            
        Returns:
            embedding向量 (1024维)
        """
        # tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            max_length=self.max_length,
            truncation=True, 
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 获取embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS] token的hidden state作为句子表示
            embeddings = outputs.last_hidden_state[:, 0, :]  # [1, 1024]
            
        return embeddings.cpu().numpy().squeeze()  # 转为numpy并去除batch维度
    
    def process_single_question(self, question_data):
        """
        处理单个问题，获取所有文本的embedding并进行mean pooling
        
        Args:
            question_data: 单个问题的数据字典
            
        Returns:
            mean_embedding: 该问题的最终语义向量
        """
        embeddings = []
        
        # 1. 获取问题文本的embedding
        if 'question_text' in question_data and question_data['question_text']:
            question_emb = self.get_text_embedding(question_data['question_text'])
            embeddings.append(question_emb)
        
        # 2. 获取所有选项的embedding
        for key, value in question_data.items():
            if key.startswith('option_') and isinstance(value, dict) and 'text' in value:
                option_text = value['text']
                if option_text:  # 确保文本不为空
                    option_emb = self.get_text_embedding(option_text)
                    embeddings.append(option_emb)

        # 3. 获取问题解析的embedding
        if "overall_analysis" in question_data and question_data["overall_analysis"]:
            question_emb_analysis = self.get_text_embedding(question_data["overall_analysis"])
            embeddings.append(question_emb_analysis)
        
        # 3. Mean pooling
        if embeddings:
            embeddings_array = np.array(embeddings)  # [num_texts, 1024]
            mean_embedding = np.mean(embeddings_array, axis=0)  # [1024]
            return mean_embedding
        else:
            # 如果没有找到任何文本，返回零向量
            return np.zeros(1024)  # BERT-base的hidden size是1024
    
    def process_questions_file(self, input_file, output_file, embedding_dim=1024, fill_value=0.0):
        """
        处理整个问题文件，按question_id顺序排列
        
        Args:
            input_file: 输入JSON文件路径
            output_file: 输出numpy文件路径
            embedding_dim: embedding维度，默认1024
            fill_value: 缺失题目的填充值，默认0.0
        """
        # 读取数据
        print("读取数据...")
        with open(input_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        print(f"总共有 {len(questions_data)} 个问题需要处理")
        
        # 获取所有question_id，并找出最大值
        question_ids = []
        for question in questions_data:
            if 'question_id' in question:
                question_ids.append(question['question_id'])
        
        if not question_ids:
            raise ValueError("找不到question_id字段！")
        
        max_question_id = max(question_ids)
        min_question_id = min(question_ids)
        
        print(f"question_id范围: {min_question_id} ~ {max_question_id}")
        print(f"实际问题数量: {len(question_ids)}")
        print(f"需要的矩阵大小: {max_question_id + 1}")
        
        # 检查缺失的question_id
        all_ids = set(range(min_question_id, max_question_id + 1))
        existing_ids = set(question_ids)
        missing_ids = all_ids - existing_ids
        
        if missing_ids:
            print(f"缺失的question_id: {sorted(list(missing_ids))}")
        else:
            print("没有缺失的question_id")
        
        # 初始化结果矩阵 [max_question_id + 1, embedding_dim]
        # 所有位置先用填充值初始化
        result_embeddings = np.full((max_question_id + 1, embedding_dim), fill_value, dtype=np.float32)
        
        # 创建question_id到数据的映射
        id_to_data = {}
        for question in questions_data:
            if 'question_id' in question:
                id_to_data[question['question_id']] = question
        
        # 处理每个问题，按question_id放到对应位置
        processed_count = 0
        
        for question_id in tqdm(sorted(existing_ids), desc="处理问题"):
            try:
                question_data = id_to_data[question_id]
                
                # 获取该问题的embedding
                question_embedding = self.process_single_question(question_data)
                
                # 放到对应位置
                result_embeddings[question_id] = question_embedding
                processed_count += 1
                
                # 可选：打印进度信息
                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count}/{len(existing_ids)} 个问题")
                    
            except Exception as e:
                print(f"处理问题 {question_id} 时出错: {e}")
                # 出错时保持填充值
                continue
        
        print(f"成功处理了 {processed_count} 个问题")
        print(f"最终矩阵形状: {result_embeddings.shape}")
        print(f"缺失问题数量: {len(missing_ids)}")
        
        # 保存为npy文件
        np.save(output_file, result_embeddings)
        print(f"结果已保存到: {output_file}")
        
        # 返回统计信息
        stats = {
            'total_matrix_size': result_embeddings.shape[0],
            'processed_questions': processed_count,
            'missing_questions': len(missing_ids),
            'missing_ids': sorted(list(missing_ids)),
            'existing_ids': sorted(list(existing_ids))
        }
        
        return result_embeddings, stats

def main(model_path, input_file, output_file, max_length=512, embedding_dim=1024, fill_value=0.0):
    """
    主函数，接受参数化的配置
    
    Args:
        model_path: BERT模型路径
        input_file: 输入JSON文件路径
        output_file: 输出numpy文件路径
        max_length: 最大序列长度，默认512
        embedding_dim: embedding维度，默认1024
        fill_value: 缺失题目的填充值，默认0.0
    """
    print(f"=== 开始处理 ===")
    print(f"模型路径: {model_path}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"最大长度: {max_length}")
    print(f"嵌入维度: {embedding_dim}")
    print(f"填充值: {fill_value}")
    print("=" * 50)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 创建embedder
    embedder = QuestionEmbedder(model_path, max_length=max_length)
    
    # 处理文件
    embeddings, stats = embedder.process_questions_file(
        input_file, output_file, 
        embedding_dim=embedding_dim, 
        fill_value=fill_value
    )
    
    print("\n=== 处理完成 ===")
    print(f"矩阵大小: {embeddings.shape}")
    print(f"成功处理: {stats['processed_questions']} 个问题")
    print(f"缺失填充: {stats['missing_questions']} 个问题")
    
    if stats['missing_ids']:
        print(f"缺失的question_id: {stats['missing_ids']}")
    
    return embeddings, stats

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='问题embedding生成工具')
    
    parser.add_argument('--model_path', type=str, default='./bge-large-zh-v1.5',
                       help='BERT模型路径')
    parser.add_argument('--input_file', type=str, default='../data/pro_emb/qid_final/7/clean_final_qid.json',
                       help='输入JSON文件路径')
    parser.add_argument('--output_file', type=str, default='../data/pro_emb/qid_final/7/question_embeddings_overall.npy',
                       help='输出numpy文件路径')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度，默认512')
    parser.add_argument('--embedding_dim', type=int, default=1024,
                       help='embedding维度，默认1024')
    parser.add_argument('--fill_value', type=float, default=0.0,
                       help='缺失题目的填充值，默认0.0')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    main(
        model_path=args.model_path,
        input_file=args.input_file,
        output_file=args.output_file,
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
        fill_value=args.fill_value
    )