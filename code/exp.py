import json
from openai import OpenAI
from typing import List, Dict, Set, Tuple

def load_dataset(file_path: str) -> List[Tuple[List[str], List[Dict]]]:
    """加载数据集，返回对话列表和对应关系标签"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [(dialog, relations) for dialog, relations in data]

def build_relation_prompt(dialog: List[str], relation_types: List[str]) -> str:
    """构建关系抽取的Prompt"""
    conversation = "\n".join(dialog)
    return f"""请从以下对话中提取人物/地点/时间之间的关系，按严格JSON格式返回：
[
  {{"head": "实体1", "tail": "实体2", "relation": "关系类型"}}, 
  ...
]

对话内容：
{conversation}

可用的关系类型列表（必须严格使用这些标签）：
{json.dumps(relation_types, ensure_ascii=False)}

要求：
1. 实体必须直接出现在对话文本中
2. 关系必须严格匹配给定类型
3. 不要添加解释性文字
4. 忽略rid、x_type、y_type等无关字段"""

def extract_relations(dialog: List[str], relation_types: List[str], model: str = "gpt-3.5-turbo") -> List[Dict]:
    """调用大模型进行关系抽取"""
    client = OpenAI(
        api_key="YOUR_API_KEY",
        base_url="https://api.chatanywhere.tech"
    )
    
    prompt = build_relation_prompt(dialog, relation_types)
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        result = json.loads(completion.choices[0].message.content)
        return result.get("relations", [])
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return []

def evaluate(predictions: Set[Tuple], gold_standards: Set[Tuple]) -> Dict[str, float]:
    """计算P/R/F1指标"""
    TP = len(predictions & gold_standards)
    FP = len(predictions - gold_standards)
    FN = len(gold_standards - predictions)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def process_dataset(dataset_path: str, output_path: str):
    """处理整个数据集并保存结果"""
    # 1. 加载数据
    dataset = load_dataset(dataset_path)
    
    # 2. 定义关系类型（根据id2rel.json生成）
    relation_types = ["per:age", "per:place_of_residence", ...]  # 替换为实际关系类型
    
    all_results = []
    metrics = {"precision": [], "recall": [], "f1": []}
    
    # 3. 处理每条对话
    for dialog, relations in dataset:
        # 转换标准格式
        gold_triples = {
            (rel["x"], rel["y"], rel["r"][0])  # 取r列表的第一个关系
            for rel in relations
        }
        
        # 调用模型预测
        pred_relations = extract_relations(dialog, relation_types)
        pred_triples = {
            (rel["head"], rel["tail"], rel["relation"])
            for rel in pred_relations
        }
        
        # 计算指标
        dialog_metrics = evaluate(pred_triples, gold_triples)
        for k in metrics:
            metrics[k].append(dialog_metrics[k])
        
        # 保存结果
        all_results.append({
            "dialog_id": len(all_results),
            "gold_triples": list(gold_triples),
            "pred_triples": list(pred_triples),
            "metrics": dialog_metrics
        })
    
    # 4. 计算全局指标
    final_metrics = {
        "precision": sum(metrics["precision"]) / len(metrics["precision"]),
        "recall": sum(metrics["recall"]) / len(metrics["recall"]),
        "f1": sum(metrics["f1"]) / len(metrics["f1"])
    }
    
    # 5. 保存结果
    with open(output_path, 'w') as f:
        json.dump({
            "results": all_results,
            "overall_metrics": final_metrics
        }, f, indent=2)
    
    return final_metrics
# 运行评估
# metrics = process_dataset(
#     dataset_path="DialogRE-v1/test.json",
#     output_path="results_v1_gpt3.5.json"
# )

# print(f"Final Metrics: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")