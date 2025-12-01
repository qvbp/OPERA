import json
import requests
import re
import logging
import os
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple  # 确保导入Tuple

class TextOnlyOptionAnalyzer:
    def __init__(self, api_key: str, base_url: str = "xxx", log_dir: str = "logs"):
        self.api_key = api_key
        self.base_url = base_url
        self.log_dir = log_dir
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 线程锁，用于保护日志输出和统计信息
        self.lock = threading.Lock()
        
    def _setup_logging(self):
        """设置日志记录"""
        # 创建日志文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.log_dir, f"analyzer_{timestamp}.log")
        
        # 配置日志格式
        log_format = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
        
        # 设置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志记录器初始化完成，日志文件：{log_filename}")
        

    
    def _intelligent_text_cleaning(self, text: str, max_length: int = 12000) -> Tuple[str, bool]:
        """
        智能文本清理（移除了图片处理部分）
        """
        if not text or len(text) <= max_length:
            return text, False
            
        original_length = len(text)
        
        # 如果文本过长，进行分段处理
        if len(text) > max_length:
            # 尝试保留重要部分
            important_patterns = [
                r'题目[：:].*?(?=选项|$)',  # 题目内容
                r'选项[A-Z][：:].*?(?=选项|$)',  # 选项内容
                r'\$[^$]+\$',  # LaTeX公式
                r'\\[a-zA-Z]+\{[^}]*\}',  # LaTeX命令
            ]
            
            important_text = ""
            for pattern in important_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                important_text += " ".join(matches) + " "
            
            if len(important_text) <= max_length:
                return important_text.strip(), True
            else:
                # 如果重要内容还是太长，截断但保留结构
                truncated = text[:max_length]
                # 尝试在合理位置截断
                for delimiter in ['\n\n', '\n', '。', '，', ' ']:
                    last_pos = truncated.rfind(delimiter)
                    if last_pos > max_length * 0.8:
                        truncated = truncated[:last_pos]
                        break
                
                return truncated + "...[内容过长已截断]", True
        
        compressed = len(text) < original_length * 0.9
        return text, compressed

    def create_analysis_prompt(self, question_data: Dict) -> Tuple[str, Dict]:
        """
        创建分析prompt - 适配新的数据格式（使用完整的problem_text）
        """
        processing_info = {
            "jy_question_id": question_data['jy_question_id'],
            "original_question_text": question_data['question_text'],
            "original_problem_text": question_data['problem_text'],
            "has_compression": False,
            "compressed_fields": []
        }
        
        # 使用智能的清理策略处理题目文本
        clean_question_text, compressed_q = self._intelligent_text_cleaning(
            question_data['question_text'], max_length=8000
        )
        if compressed_q:
            processing_info["has_compression"] = True
            processing_info["compressed_fields"].append("question_text")
        
        # 使用智能的清理策略处理完整题目（包含选项）
        clean_problem_text, compressed_p = self._intelligent_text_cleaning(
            question_data['problem_text'], max_length=12000
        )
        if compressed_p:
            processing_info["has_compression"] = True
            processing_info["compressed_fields"].append("problem_text")
        
        # 获取题目的知识点列表
        available_kcs = question_data.get('kc_texts', [])
        kcs_str = '", "'.join(available_kcs)
        
        # 获取正确答案和选项数量
        correct_answer = question_data.get('correct_answer', 'A')
        options_num = question_data.get('options_num', 4)
        
        processing_info["clean_question_text"] = clean_question_text
        processing_info["clean_problem_text"] = clean_problem_text
        processing_info["options_num"] = options_num
        
        # 动态生成选项分析结构
        option_letters = [chr(65 + i) for i in range(options_num)]  # A, B, C, D, E...
        
        # 构建选项分析的JSON模板
        option_analysis_template = {}
        for option_letter in option_letters:
            is_correct = option_letter == correct_answer
            
            option_template = {
                "involved_kcs": ["从题目知识点中选择的具体知识点"],
                "mastery_level": {
                    "具体知识点名称": "掌握程度(0-4整数，根据涉及的知识点动态确定)"
                },
                "reasoning": "选择此选项的学生认知分析",
                "is_correct": is_correct
            }
            
            if not is_correct:
                option_template["error_analysis"] = {
                    "error_types": [
                        {
                            "type": "错误类型名称",
                            "description": "具体错误描述", 
                            "suggestion": "针对该错误类型的具体改进建议"
                        }
                    ],
                    "formatted_analysis": "错误类型:[具体的错误类型名称];错误描述:[具体的错误描述]；建议:[针对该错误类型的具体改进建议]"
                }
            
            option_analysis_template[f"option_{option_letter}"] = option_template
        
        # 将模板转换为JSON字符串，用于在prompt中展示
        import json
        options_json_template = json.dumps(option_analysis_template, ensure_ascii=False, indent=12)
        
        prompt = f"""
    你是一个教育心理学和学科教学专家，请分析以下数学选择题，重点分析学生的错误类型。

    【重要提示】
    - 题目和选项中可能包含LaTeX数学公式，请正确理解和分析
    - 根据知识点"{', '.join(available_kcs)}"来分析题目要求
    - **正确答案已经提供给你，请基于这个正确答案进行分析**
    - 重点分析错误选项的错误原因和类型

    【题目信息】
    题目ID: {question_data['jy_question_id']}
    涉及知识点: {', '.join(available_kcs)}
    **正确答案: {correct_answer}**
    **选项数量: {options_num}个**

    【完整题目（包含所有选项）】
    {clean_problem_text}

    【分析任务】
    1. **基于已知正确答案({correct_answer})**: 请解释为什么这个选项是正确的
    2. **错误类型分析**: 对每个错误选项进行详细的错误类型分析，分析学生为什么会选择错误选项

    【错误类型分析要求】
    参考以下错误分析框架：

    ## 数学选择题错误分析任务
    你的任务是分析学生在数学选择题中的错误类型。

    ### 分析步骤
    1. 仔细阅读题目内容，理解题目要求
    2. **基于已提供的正确答案({correct_answer})分析正确解题过程**
    3. 对比其他选项与正确答案的差异
    4. 分析学生为什么会选择错误选项的具体原因
    5. 将错误原因归类到对应的错误类型
    6. 针对每种错误类型提供具体的改进建议

    ### 常见错误类型参考
    - 计算错误：基本运算出错
    - 理解错误：题意理解偏差
    - 概念混淆：相关概念辨析不清
    - 方法错误：解题方法选择不当
    - 步骤遗漏：解题过程不完整
    - 单位错误：单位换算或标注错误
    - 抄写错误：题目或运算过程中的抄写失误
    - 审题错误：题目条件遗漏或误读
    - 格式错误：答案格式或解题步骤书写不规范

    【每个选项的分析要求】
    对每个选项分析以下内容：
    - **涉及知识点**: 从以下知识点中选择：["{kcs_str}"]
    - **知识掌握程度评估**: 如果学生选择此选项，反映其对相关知识点的掌握程度（每个涉及的知识点都要评估）
    - **选择原因分析**: 学生为什么会选择这个选项（正确推理/错误理解/猜测等）
    - **错误类型分析**（仅针对错误选项）: 按照指定格式分析错误类型和建议

    **掌握程度评分标准 (0-4级):**
    - 4级(完全掌握): 选择正确选项且理由充分，完全理解概念和应用
    - 3级(较好掌握): 基本理解正确但可能有细节疏漏，或正确选项但理由不够完整
    - 2级(部分掌握): 有一定理解但存在明显误解，导致选择错误
    - 1级(初步了解): 对概念有模糊认识但理解有严重偏差
    - 0级(完全不掌握): 完全不理解相关概念，随机选择或基于错误理解

    **注意**: 每个选项可能涉及1个或多个知识点，请根据实际情况在mastery_level中列出所有相关知识点及其掌握程度

    【输出格式】
    请严格按照以下JSON格式输出：

    {{
        "jy_question_id": "{question_data['jy_question_id']}",
        "correct_answer": "{correct_answer}",
        "correct_answer_explanation": "基于正确答案{correct_answer}的详细解题过程和原理解释",
        "analysis": {options_json_template},
        "overall_analysis": "整体题目的知识点考查分析和难度评估"
    }}

    注意事项：
    1. 请确保JSON格式正确，可以被Python json.loads()解析
    2. correct_answer字段必须输出：{correct_answer}
    3. involved_kcs必须从题目给定的知识点列表中选择：["{kcs_str}"]
    4. 掌握程度评分要合理，体现选项选择与知识掌握的对应关系
    5. 错误分析只针对错误选项，格式要求：错误类型:[类型名称]；错误描述:[具体描述]；建议:[改进建议]
    6. 必须包含question_id字段用于结果对应
    7. error_analysis中的formatted_analysis字段要严格按照指定格式
    8. 基于已提供的正确答案({correct_answer})进行分析，无需重新推理正确答案
    9. 题目总共有{options_num}个选项，请对每个选项都进行分析
    10. mastery_level中的数值必须是0-4之间的整数，不要用字符串
    11. 每个选项在mastery_level中应列出该选项实际涉及的所有知识点（可能是1个、2个或多个），知识点名称必须与involved_kcs中的一致
    """
        
        return prompt, processing_info
    
    def analyze_single_question_with_retry(self, question_data: Dict, max_retries: int = 5) -> Dict:
        """
        简化的重试机制 - 针对超时进行特殊处理
        """
        for attempt in range(max_retries + 1):
            try:
                result = self.analyze_single_question(question_data)
                
                if result['success']:
                    if attempt > 0:
                        with self.lock:
                            self.logger.info(f"题目{question_data['jy_question_id']} 在第{attempt + 1}次尝试成功")
                    return result
                else:
                    # 如果是超时错误，增加等待时间
                    if "超时" in result.get('error', '') or "timeout" in result.get('error', '').lower():
                        if attempt < max_retries:
                            wait_time = 5 + attempt * 2  # 5, 7, 9, 11, 13秒递增等待
                            with self.lock:
                                self.logger.warning(f"题目{question_data['jy_question_id']} 第{attempt + 1}次超时，等待{wait_time}秒后重试")
                            time.sleep(wait_time)
                        else:
                            with self.lock:
                                self.logger.error(f"题目{question_data['jy_question_id']} 达到最大重试次数，仍然超时")
                            return result
                    else:
                        # 其他错误，正常重试
                        if attempt < max_retries:
                            with self.lock:
                                self.logger.warning(f"题目{question_data['jy_question_id']} 第{attempt + 1}次尝试失败: {result.get('error', 'Unknown error')}")
                            time.sleep(2)
                        else:
                            with self.lock:
                                self.logger.error(f"题目{question_data['jy_question_id']} 达到最大重试次数")
                            return result
                            
            except Exception as e:
                if attempt < max_retries:
                    with self.lock:
                        self.logger.warning(f"题目{question_data['jy_question_id']} 第{attempt + 1}次尝试异常: {str(e)}")
                    time.sleep(2)
                else:
                    with self.lock:
                        self.logger.error(f"题目{question_data['jy_question_id']} 达到最大重试次数，异常: {str(e)}")
                    return {
                        "jy_question_id": question_data['jy_question_id'],
                        "success": False,
                        "error": f"最终异常: {str(e)}",
                        "processing_info": {},
                        "raw_response": None,
                        "prompt_length": 0
                    }
        
        return {
            "jy_question_id": question_data['jy_question_id'],
            "success": False,
            "error": "未知错误",
            "processing_info": {},
            "raw_response": None,
            "prompt_length": 0
        }
    
    def analyze_single_question(self, question_data: Dict) -> Dict:
        """
        简化版分析方法 - 主要通过延长超时解决长prompt问题
        """
        prompt, processing_info = self.create_analysis_prompt(question_data)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 估算prompt长度并设置对应的超时时间
        prompt_length = len(prompt)
        estimated_tokens = len(prompt) // 2  # 粗略估算：2字符≈1token
        
        # 根据长度设置超时时间和max_tokens
        if estimated_tokens > 25000:
            timeout = 300  # 5分钟 - 超长prompt
            max_tokens = 8000
            with self.lock:
                self.logger.info(f"题目{question_data['jy_question_id']} 检测到超长prompt ({estimated_tokens} tokens)，设置5分钟超时")
        elif estimated_tokens > 15000:
            timeout = 180  # 3分钟 - 长prompt  
            max_tokens = 6000
            with self.lock:
                self.logger.debug(f"题目{question_data['jy_question_id']} 检测到长prompt ({estimated_tokens} tokens)，设置3分钟超时")
        elif estimated_tokens > 10000:
            timeout = 120  # 2分钟 - 中等prompt
            max_tokens = 4000
        else:
            timeout = 120   # 1.5分钟 - 正常prompt
            max_tokens = 3000
        
        # payload = {
        #     "model": "Qwen3-235B-A22B", 
        #     "messages": [
        #         {
        #             "role": "user", 
        #             "content": prompt
        #         }
        #     ],
        #     "temperature": 0.1,
        #     "max_tokens": max_tokens
        # }
        
        payload = {
            "model": "qwen3-235b-a22b", #"Qwen3-235B-A22B", 
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }

        try:
            # 使用动态调整的长超时时间
            response = requests.post(
                f"{self.base_url}/chat/completions", 
                headers=headers, 
                json=payload, 
                timeout=timeout
            )
            response.raise_for_status()
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API请求失败，状态码: {response.status_code}"
                with self.lock:
                    self.logger.error(f"题目{question_data['jy_question_id']} - {error_msg}")
                return {
                    "jy_question_id": question_data['jy_question_id'],
                    "success": False,
                    "error": error_msg,
                    "processing_info": processing_info,
                    "raw_response": response.text,
                    "prompt_length": len(prompt)
                }
            
            # 解析响应
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"API响应JSON解析失败: {str(e)}"
                with self.lock:
                    self.logger.error(f"题目{question_data['jy_question_id']} - {error_msg}")
                return {
                    "jy_question_id": question_data['jy_question_id'],
                    "success": False,
                    "error": error_msg,
                    "processing_info": processing_info,
                    "raw_response": response.text[:1000],
                    "prompt_length": len(prompt)
                }
            
            # 检查响应结构
            if 'choices' not in result or not result['choices']:
                error_msg = f"API响应格式异常"
                with self.lock:
                    self.logger.error(f"题目{question_data['jy_question_id']} - {error_msg}")
                return {
                    "jy_question_id": question_data['jy_question_id'],
                    "success": False,
                    "error": error_msg,
                    "processing_info": processing_info,
                    "raw_response": str(result),
                    "prompt_length": len(prompt)
                }
            
            # 获取内容
            content = result['choices'][0]['message']['content']
            
            if not content or not content.strip():
                error_msg = "API返回内容为空"
                with self.lock:
                    self.logger.error(f"题目{question_data['jy_question_id']} - {error_msg}")
                return {
                    "jy_question_id": question_data['jy_question_id'],
                    "success": False,
                    "error": error_msg,
                    "processing_info": processing_info,
                    "raw_response": content,
                    "prompt_length": len(prompt)
                }
            
            try:
                # JSON解析
                analysis_result = self.parse_response_json(content)
                
                # 基本验证
                if not isinstance(analysis_result, dict):
                    error_msg = f"解析结果不是字典类型"
                    return {
                        "jy_question_id": question_data['jy_question_id'],
                        "success": False,
                        "error": error_msg,
                        "processing_info": processing_info,
                        "raw_response": content,
                        "prompt_length": len(prompt)
                    }
                
                # 确保包含question_id
                if 'jy_question_id' not in analysis_result:
                    analysis_result['jy_question_id'] = question_data['jy_question_id']
                
                with self.lock:
                    self.logger.info(f"✓ 题目{question_data['jy_question_id']} 成功处理 (timeout={timeout}s, tokens≈{estimated_tokens})")
                
                return {
                    "jy_question_id": question_data['jy_question_id'],
                    "success": True,
                    "analysis": analysis_result,
                    "processing_info": processing_info,
                    "raw_response": content,
                    "prompt_length": len(prompt)
                }
                
            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f"内容JSON解析失败: {str(e)}"
                with self.lock:
                    self.logger.error(f"题目{question_data['jy_question_id']} - {error_msg}")
                    self.logger.error(f"原始响应内容前500字符: {content[:500]}")
                return {
                    "jy_question_id": question_data['jy_question_id'],
                    "success": False,
                    "error": error_msg,
                    "processing_info": processing_info,
                    "raw_response": content,
                    "prompt_length": len(prompt)
                }
                    
        except requests.Timeout:
            error_msg = f"请求超时 (等待了{timeout}秒, prompt长度≈{estimated_tokens} tokens)"
            with self.lock:
                self.logger.error(f"题目{question_data['jy_question_id']} - {error_msg}")
            return {
                "jy_question_id": question_data['jy_question_id'],
                "success": False,
                "error": error_msg,
                "processing_info": processing_info,
                "raw_response": None,
                "prompt_length": len(prompt)
            }
        except requests.RequestException as e:
            error_msg = f"网络请求异常: {str(e)}"
            with self.lock:
                self.logger.error(f"题目{question_data['jy_question_id']} - {error_msg}")
            return {
                "jy_question_id": question_data['jy_question_id'],
                "success": False,
                "error": error_msg,
                "processing_info": processing_info,
                "raw_response": None,
                "prompt_length": len(prompt)
            }
        except Exception as e:
            error_msg = f"未知异常: {str(e)}"
            with self.lock:
                self.logger.error(f"题目{question_data['jy_question_id']} - {error_msg}")
            return {
                "jy_question_id": question_data['jy_question_id'],
                "success": False,
                "error": error_msg,
                "processing_info": processing_info,
                "raw_response": None,
                "prompt_length": len(prompt)
            }
            
    def parse_response_json(self, content: str) -> dict:
        """
        改进的JSON解析方法，能处理<think>标签和其他格式
        """
        
        # 首先检查内容是否为空或None
        if not content:
            raise ValueError("API返回内容为空")
        
        # 去除首尾空白字符
        content = content.strip()
        
        # 再次检查处理后的内容
        if not content:
            raise ValueError("API返回内容去除空白后为空")
        
        # 记录原始内容的前200个字符用于调试
        with self.lock:
            self.logger.debug(f"原始响应内容前200字符: {content[:200]}")
        
        # 处理<think>标签 - 这是关键的新增处理！
        if content.startswith('<think>'):
            with self.lock:
                self.logger.debug("检测到<think>标签，尝试提取JSON内容")
            
            # 查找</think>标签
            think_end = content.find('</think>')
            if think_end != -1:
                # 提取</think>之后的内容
                json_content = content[think_end + 8:].strip()  # 8 = len('</think>')
                with self.lock:
                    self.logger.debug(f"提取<think>后的内容前200字符: {json_content[:200]}")
            else:
                # 如果没有找到</think>，可能内容被截断了，尝试查找JSON开始
                json_start = content.find('{')
                if json_start != -1:
                    json_content = content[json_start:].strip()
                    with self.lock:
                        self.logger.debug("没有找到</think>，但找到了JSON开始位置")
                else:
                    # 如果连JSON开始都找不到，记录错误并抛出异常
                    with self.lock:
                        self.logger.error(f"检测到<think>标签但无法找到JSON内容，完整内容: {content}")
                    raise ValueError("检测到<think>标签但无法找到JSON内容")
            
            # 使用提取的内容作为待解析的JSON
            content = json_content
        
        # 处理其他可能的thinking格式
        elif '<thinking>' in content:
            with self.lock:
                self.logger.debug("检测到<thinking>标签，尝试提取JSON内容")
            
            thinking_end = content.find('</thinking>')
            if thinking_end != -1:
                json_content = content[thinking_end + 11:].strip()  # 11 = len('</thinking>')
                content = json_content
            else:
                # 查找JSON开始位置
                json_start = content.find('{')
                if json_start != -1:
                    content = content[json_start:].strip()
        
        # 现在尝试提取JSON格式的内容
        json_str = None
        
        # 方法1: 查找 ```json 代码块
        if '```json' in content:
            start = content.find('```json') + 7
            end = content.find('```', start)
            if end != -1:
                json_str = content[start:end].strip()
                with self.lock:
                    self.logger.debug("使用```json代码块解析")
        
        # 方法2: 查找普通 ``` 代码块
        elif '```' in content:
            start = content.find('```') + 3
            end = content.find('```', start)
            if end != -1:
                json_str = content[start:end].strip()
                with self.lock:
                    self.logger.debug("使用```代码块解析")
        
        # 方法3: 尝试查找JSON对象（以{开头}结尾）
        elif '{' in content and '}' in content:
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end].strip()
            with self.lock:
                self.logger.debug("使用{}包围的内容解析")
        
        # 方法4: 直接使用全部内容
        else:
            json_str = content
            with self.lock:
                self.logger.debug("使用全部内容解析")
        
        # 检查提取的JSON字符串
        if not json_str:
            with self.lock:
                self.logger.error(f"无法从响应中提取JSON内容。原始内容: {content[:500]}")
            raise ValueError(f"无法从响应中提取JSON内容。原始内容: {content[:500]}")
        
        # 记录提取的JSON字符串
        with self.lock:
            self.logger.debug(f"提取的JSON字符串前200字符: {json_str[:200]}")
        
        # 尝试解析JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # 提供更详细的错误信息
            error_msg = f"JSON解析失败: {str(e)}\n"
            error_msg += f"错误位置: 第{e.lineno}行, 第{e.colno}列\n"
            error_msg += f"原始内容长度: {len(content)}\n"
            error_msg += f"提取的JSON长度: {len(json_str)}\n"
            error_msg += f"提取的JSON内容: {json_str[:1000]}\n"
            error_msg += f"原始内容: {content[:1000]}"
            
            with self.lock:
                self.logger.error(error_msg)
            
            raise ValueError(error_msg)
    
    def _save_single_result_file(self, result: Dict, output_dir: str) -> bool:
        """
        将单个结果保存为独立的JSON文件
        """
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 使用question_id作为文件名
            filename = f"{result['jy_question_id']}.json"
            filepath = os.path.join(output_dir, filename)
            
            # 检查文件是否已存在
            if os.path.exists(filepath):
                with self.lock:
                    self.logger.debug(f"题目 {result['jy_question_id']} 文件已存在，跳过保存")
                return False
            
            # 保存文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            with self.lock:
                self.logger.debug(f"成功保存题目 {result['jy_question_id']} 到文件 {filepath}")
            return True
            
        except Exception as e:
            with self.lock:
                self.logger.error(f"保存题目 {result['jy_question_id']} 失败: {str(e)}")
            return False
    
    def _load_existing_results_from_dir(self, output_dir: str) -> set:
        """
        从输出目录加载已处理的question_id集合
        """
        processed_ids = set()
        
        if not os.path.exists(output_dir):
            self.logger.info(f"输出目录 {output_dir} 不存在，从头开始处理")
            return processed_ids
        
        try:
            # 扫描目录中的所有JSON文件
            for filename in os.listdir(output_dir):
                if filename.endswith('.json'):
                    try:
                        # 从文件名提取question_id
                        jy_question_id = int(filename.replace('.json', ''))
                        processed_ids.add(jy_question_id)
                    except ValueError:
                        # 文件名不是纯数字，跳过
                        continue
            
            self.logger.info(f"从目录 {output_dir} 加载已处理的题目数: {len(processed_ids)}")
            
        except Exception as e:
            self.logger.warning(f"加载已有结果失败: {str(e)}，从头开始处理")
            
        return processed_ids
    
    def batch_analyze_multithreaded(self, data_file: str, output_dir: str, 
                                  num_threads: int = 8, max_retries: int = 5, 
                                  start_idx: int = 0):
        """
        多线程批量分析方法 - 修复版：返回所有结果，只保存成功的
        
        Args:
            data_file: 输入数据文件
            output_dir: 输出结果目录（只保存成功的JSON文件）
            num_threads: 线程数量，默认8个
            max_retries: 最大重试次数，默认5次
            start_idx: 开始索引
            
        Returns:
            List[Dict]: 所有处理结果（包括成功和失败的）
        """
        self.logger.info(f"开始多线程批量分析，数据文件: {data_file}")
        self.logger.info(f"线程数: {num_threads}, 最大重试次数: {max_retries}")
        
        # 读取题目数据
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            self.logger.info(f"成功加载 {len(questions)} 个题目")
        except Exception as e:
            self.logger.error(f"读取数据文件失败: {str(e)}")
            raise
        
        # 加载已处理的题目ID
        processed_ids = self._load_existing_results_from_dir(output_dir)
        
        # 过滤未处理的题目
        pending_questions = []
        for i, question in enumerate(questions[start_idx:], start_idx):
            if question['jy_question_id'] not in processed_ids:
                pending_questions.append((i, question))
        
        total = len(questions)
        pending_count = len(pending_questions)
        
        self.logger.info(f"总题目数: {total}")
        self.logger.info(f"已处理: {len(processed_ids)}")
        self.logger.info(f"待处理: {pending_count}")
        
        if pending_count == 0:
            self.logger.info("所有题目都已处理完成")
            # 如果没有待处理的，直接从目录收集结果（只有成功的）
            return self.collect_results_from_dir(output_dir)
        
        # 统计信息（线程安全）
        stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "correct_answers": {"A": 0, "B": 0, "C": 0, "D": 0, "Unknown": 0}
        }
        
        # 存储所有结果的列表（线程安全）- 包括成功和失败的
        all_results = []
        results_lock = threading.Lock()
        
        def process_question(question_info):
            """处理单个题目的函数"""
            idx, question = question_info
            
            # 分析题目（带重试）
            result = self.analyze_single_question_with_retry(question, max_retries)
            
            # 将结果添加到总列表（包括失败的）
            with results_lock:
                all_results.append(result)
            
            # 只保存成功的结果到output_dir
            if result['success']:
                save_success = self._save_single_result_file(result, output_dir)
                with self.lock:
                    self.logger.debug(f"题目{question['jy_question_id']} 成功分析并保存到文件")
            else:
                with self.lock:
                    self.logger.debug(f"题目{question['jy_question_id']} 分析失败，未保存文件但已记录到结果列表")
            
            # 更新统计（线程安全）
            with self.lock:
                stats["processed"] += 1
                
                if result['success']:
                    stats["success"] += 1
                    correct_answer = result.get('analysis', {}).get('correct_answer', 'Unknown')
                    if correct_answer in stats["correct_answers"]:
                        stats["correct_answers"][correct_answer] += 1
                    else:
                        stats["correct_answers"]['Unknown'] += 1
                        
                    compression_info = result['processing_info'].get('has_compression', False)
                    self.logger.info(f"✓ 题目{question['jy_question_id']} ({idx+1}/{total}) - 正确答案: {correct_answer} - 压缩: {compression_info} - Prompt长度: {result['prompt_length']}")
                else:
                    stats["failed"] += 1
                    stats["correct_answers"]['Unknown'] += 1
                    self.logger.error(f"✗ 题目{question['jy_question_id']} ({idx+1}/{total}) - 分析失败: {result['error']}")
                
                # 每10个题目输出一次统计
                if stats["processed"] % 10 == 0:
                    success_rate = stats['success'] / stats['processed'] * 100
                    self.logger.info(f"当前进度: {stats['processed']}/{pending_count} - 成功{stats['success']}, 失败{stats['failed']} - 成功率: {success_rate:.1f}%")
            
            return result
        
        # 使用线程池执行
        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # 提交所有任务
                future_to_question = {
                    executor.submit(process_question, question_info): question_info[1]['jy_question_id'] 
                    for question_info in pending_questions
                }
                
                # 等待所有任务完成
                for future in as_completed(future_to_question):
                    jy_question_id = future_to_question[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        with self.lock:
                            self.logger.error(f"题目 {jy_question_id} 处理异常: {str(e)}")
                        
        except KeyboardInterrupt:
            self.logger.warning("用户中断处理")
            raise
        except Exception as e:
            self.logger.error(f"多线程处理过程中发生错误: {str(e)}")
            raise
        
        # 合并新结果和已有结果
        existing_results = self.collect_results_from_dir(output_dir)
        
        # 创建已有结果的ID集合，避免重复
        existing_ids = {r['jy_question_id'] for r in existing_results}
        
        # 只添加新的结果（避免重复）
        final_results = existing_results.copy()
        for result in all_results:
            if result['jy_question_id'] not in existing_ids:
                final_results.append(result)
        
        # 输出最终统计
        self.logger.info("="*50)
        self.logger.info("多线程分析完成！")
        self.logger.info(f"本次运行统计:")
        self.logger.info(f"- 处理题目: {stats['processed']}")
        self.logger.info(f"- 成功分析: {stats['success']}")
        self.logger.info(f"- 分析失败: {stats['failed']}")
        self.logger.info(f"- 成功率: {stats['success']/stats['processed']*100:.1f}%" if stats['processed'] > 0 else "- 成功率: 0%")
        self.logger.info(f"- 正确答案分布: {stats['correct_answers']}")
        self.logger.info(f"- 使用线程数: {num_threads}")
        self.logger.info(f"- 最大重试次数: {max_retries}")
        self.logger.info(f"- 返回结果总数: {len(final_results)} (包括所有成功和失败的)")
        self.logger.info("="*50)
        
        return final_results
    
    def collect_results_from_dir(self, output_dir: str) -> List[Dict]:
        """
        从输出目录收集所有结果文件
        """
        results = []
        
        if not os.path.exists(output_dir):
            self.logger.warning(f"输出目录 {output_dir} 不存在")
            return results
        
        try:
            for filename in os.listdir(output_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(output_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"读取结果文件 {filepath} 失败: {str(e)}")
            
            self.logger.info(f"从目录 {output_dir} 收集到 {len(results)} 个结果")
            
        except Exception as e:
            self.logger.error(f"收集结果失败: {str(e)}")
        
        return results

    def export_results_with_original_mapping(self, results: List[Dict], output_file: str):
        """
        导出结果，包含错误分析
        """
        self.logger.info(f"开始导出结果到 {output_file}")
        
        # 分类统计
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        # 正确答案分布统计
        correct_answer_distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "Unknown": 0}
        for result in successful_results:
            correct_answer = result.get('analysis', {}).get('correct_answer', 'Unknown')
            if correct_answer in correct_answer_distribution:
                correct_answer_distribution[correct_answer] += 1
            else:
                correct_answer_distribution['Unknown'] += 1
        
        # 错误类型统计
        error_type_stats = {}
        for result in successful_results:
            if 'analysis' in result and 'analysis' in result['analysis']:
                for option_key, option_data in result['analysis']['analysis'].items():
                    if not option_data.get('is_correct', True) and 'error_analysis' in option_data:
                        error_types = option_data['error_analysis'].get('error_types', [])
                        for error_type in error_types:
                            error_name = error_type.get('type', 'Unknown')
                            error_type_stats[error_name] = error_type_stats.get(error_name, 0) + 1
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_questions": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(failed_results),
            "correct_answer_distribution": correct_answer_distribution,
            "error_type_distribution": error_type_stats,
            "compression_stats": {
                "total_compressed": sum(1 for r in results if r.get('processing_info', {}).get('has_compression', False)),
                "compression_fields": {}
            },
            "results": []
        }
        
        # 统计压缩信息
        for result in results:
            if result.get('processing_info', {}).get('has_compression', False):
                for field in result['processing_info']['compressed_fields']:
                    export_data["compression_stats"]["compression_fields"][field] = \
                        export_data["compression_stats"]["compression_fields"].get(field, 0) + 1
        
        # 整理结果
        for result in results:
            # 所有题目都是正常处理的文本题
            export_result = {
                "jy_question_id": result['jy_question_id'],
                "success": result['success'],
                "correct_answer": result.get('analysis', {}).get('correct_answer', None),
                "correct_answer_explanation": result.get('analysis', {}).get('correct_answer_explanation', None),
                "original_question": result['processing_info'].get('original_question_text', ''),
                "original_options": result['processing_info'].get('original_options', {}),
                "processed_question": result['processing_info'].get('clean_question_text', ''),
                "processed_options": result['processing_info'].get('clean_options', {}),
                "has_text_compression": result['processing_info'].get('has_compression', False),
                "compressed_fields": result['processing_info'].get('compressed_fields', []),
                "analysis": result.get('analysis', None),
                "error": result.get('error', None)
            }
            export_data["results"].append(export_result)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"结果已导出到 {output_file}")
            self.logger.info(f"导出统计信息：")
            self.logger.info(f"- 总题目数：{export_data['total_questions']}")
            self.logger.info(f"- 成功分析：{export_data['successful_analyses']}")
            self.logger.info(f"- 分析失败：{export_data['failed_analyses']}")
            self.logger.info(f"- 文本压缩：{export_data['compression_stats']['total_compressed']}")
            self.logger.info(f"- 正确答案分布：{export_data['correct_answer_distribution']}")
            self.logger.info(f"- 错误类型分布：{export_data['error_type_distribution']}")
        except Exception as e:
            self.logger.error(f"导出结果失败: {str(e)}")
            raise
            
        return export_data

    def export_failed_questions(self, results: List[Dict], output_file: str) -> Dict:
        """
        导出处理失败的题目ID和错误信息
        
        Args:
            results: 所有结果列表
            output_file: 输出文件路径
            
        Returns:
            包含失败统计信息的字典
        """
        self.logger.info(f"开始导出失败题目信息到 {output_file}")
        
        # 筛选失败的结果
        failed_results = [r for r in results if not r['success']]
        
        # 按错误类型分类
        error_categories = {}
        failed_question_ids = []
        
        for result in failed_results:
            jy_question_id = result['jy_question_id']
            error_msg = result.get('error', 'Unknown error')
            
            failed_question_ids.append(jy_question_id)
            
            # 简化错误信息进行分类
            if 'JSON解析失败' in error_msg:
                category = 'JSON解析错误'
            elif '异常' in error_msg or 'Exception' in error_msg:
                category = 'API异常'
            elif '超时' in error_msg or 'timeout' in error_msg:
                category = '请求超时'
            elif '最终异常' in error_msg:
                category = '重试后仍失败'
            else:
                category = '其他错误'
            
            if category not in error_categories:
                error_categories[category] = []
            
            error_categories[category].append({
                'jy_question_id': jy_question_id,
                'error': error_msg,
                'raw_response': result.get('raw_response', None),
                'prompt_length': result.get('prompt_length', 0)
            })
        
        # 构建导出数据
        failed_data = {
            "export_time": datetime.now().isoformat(),
            "total_failed": len(failed_results),
            "failed_question_ids": sorted(failed_question_ids),  # 排序后的失败ID列表
            "error_statistics": {
                category: len(questions) for category, questions in error_categories.items()
            },
            "error_details_by_category": error_categories,
            "detailed_failed_list": [
                {
                    "jy_question_id": result['jy_question_id'],
                    "error": result.get('error', 'Unknown error'),
                    "prompt_length": result.get('prompt_length', 0),
                    "has_raw_response": bool(result.get('raw_response'))
                }
                for result in failed_results
            ]
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(failed_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"失败题目信息已导出到 {output_file}")
            self.logger.info(f"失败题目统计：")
            self.logger.info(f"- 总失败数：{failed_data['total_failed']}")
            self.logger.info(f"- 错误类型分布：{failed_data['error_statistics']}")
            
            # 单独保存失败的question_id列表（纯文本格式，方便查看）
            failed_ids_txt = output_file.replace('.json', '_ids.txt')
            with open(failed_ids_txt, 'w', encoding='utf-8') as f:
                f.write("处理失败的题目ID列表:\n")
                f.write("=" * 30 + "\n")
                for qid in sorted(failed_question_ids):
                    f.write(f"{qid}\n")
                f.write("=" * 30 + "\n")
                f.write(f"总计: {len(failed_question_ids)} 个失败题目\n")
            
            self.logger.info(f"失败题目ID列表已保存到 {failed_ids_txt}")
            
        except Exception as e:
            self.logger.error(f"导出失败题目信息失败: {str(e)}")
            raise
            
        return failed_data
    
    def get_processing_summary(self, results: List[Dict]) -> Dict:
        """
        获取处理结果的详细统计摘要
        """
        total = len(results)
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        # 成功率统计
        success_rate = len(successful) / total * 100 if total > 0 else 0
        
        # 正确答案分布
        answer_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "Unknown": 0}
        for result in successful:
            answer = result.get('analysis', {}).get('correct_answer', 'Unknown')
            if answer in answer_dist:
                answer_dist[answer] += 1
            else:
                answer_dist['Unknown'] += 1
        
        # 压缩统计
        compressed_count = sum(1 for r in results if r.get('processing_info', {}).get('has_compression', False))
        
        # 失败原因统计
        failure_reasons = {}
        for result in failed:
            error = result.get('error', 'Unknown')
            if 'JSON解析失败' in error:
                key = 'JSON解析失败'
            elif '异常' in error:
                key = 'API异常'
            elif '超时' in error:
                key = '请求超时'
            else:
                key = '其他错误'
            failure_reasons[key] = failure_reasons.get(key, 0) + 1
        
        return {
            "总题目数": total,
            "成功分析": len(successful),
            "分析失败": len(failed),
            "成功率": f"{success_rate:.1f}%",
            "正确答案分布": answer_dist,
            "文本压缩数量": compressed_count,
            "失败原因分布": failure_reasons,
            "失败题目ID": sorted([r['jy_question_id'] for r in failed])
        }

    def get_original_question_info(self, results: List[Dict], jy_question_id: int) -> Dict:
        """根据question_id获取原始题目信息和分析结果"""
        for result in results:
            if result['jy_question_id'] == jy_question_id:
                return {
                    "jy_question_id": jy_question_id,
                    "correct_answer": result.get('analysis', {}).get('correct_answer', None),
                    "correct_answer_explanation": result.get('analysis', {}).get('correct_answer_explanation', None),
                    "original_data": {
                        "question_text": result['processing_info']['original_question_text'],
                        "options": result['processing_info']['original_options']
                    },
                    "processed_data": {
                        "question_text": result['processing_info']['clean_question_text'],
                        "options": result['processing_info']['clean_options']
                    },
                    "compression_info": {
                        "has_compression": result['processing_info']['has_compression'],
                        "compressed_fields": result['processing_info']['compressed_fields']
                    },
                    "analysis_result": result.get('analysis', None),
                    "success": result['success']
                }
        return None


# 使用示例
if __name__ == "__main__":
    # 初始化分析器（指定日志目录）

    analyzer = TextOnlyOptionAnalyzer(
        base_url="xxxx",
        api_key="xxx",
        log_dir="analyzer_logs_test"  # 指定日志目录
    )

    
    try:
        # 多线程批量分析（每个题目保存为单独文件）
        # 现在直接使用返回的结果，包含所有成功和失败的题目
        results = analyzer.batch_analyze_multithreaded(
            data_file="../get_analysis/45/result/failed_0723.json",
            output_dir="../get_analysis/45/results_individual_0723_v1",
            num_threads=8,  # 可以调整线程数量：8, 16, 32等
            max_retries=8,   # 最大重试次数
            start_idx=0
        )
        
        # 不再需要重新收集结果，直接使用返回的results
        # results = analyzer.collect_results_from_dir(...)  # 删除这行！
        
        analyzer.logger.info(f"获得完整结果: 总数{len(results)}, 成功{len([r for r in results if r['success']])}, 失败{len([r for r in results if not r['success']])}")
        
        # 导出失败题目的详细信息
        failed_data = analyzer.export_failed_questions(
            results, "../get_analysis/45/result/failed_questions_0723_v1.json"
        )
        
        # 获取处理摘要
        summary = analyzer.get_processing_summary(results)
        
        # 查看特定题目的信息
        if results:
            question_info = analyzer.get_original_question_info(results, jy_question_id=742)
            if question_info:
                analyzer.logger.info("="*30 + " 题目信息 " + "="*30)
                analyzer.logger.info(f"题目ID: {question_info['jy_question_id']}")
                analyzer.logger.info(f"正确答案: {question_info['correct_answer']}")
                analyzer.logger.info(f"分析成功: {question_info['success']}")
        
        # 输出详细统计
        analyzer.logger.info("="*30 + " 最终统计 " + "="*30)
        for key, value in summary.items():
            if key != "失败题目ID":  # 失败ID列表太长，不在这里显示
                analyzer.logger.info(f"- {key}: {value}")
        
        # 如果有失败的题目，显示失败ID
        if summary["失败题目ID"]:
            analyzer.logger.info(f"- 失败题目ID数量: {len(summary['失败题目ID'])}")
            analyzer.logger.info(f"- 失败题目ID: {summary['失败题目ID']}")
            analyzer.logger.info(f"- 失败题目详细信息已保存到: failed_questions_0723_v1.json")
            
    except KeyboardInterrupt:
        analyzer.logger.info("程序被用户中断")
    except Exception as e:
        analyzer.logger.error(f"程序执行出错: {str(e)}")
        raise