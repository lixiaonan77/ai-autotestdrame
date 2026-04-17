"""
RAG系统质量自动化测试
使用RAGAS评估指标 + pytest
"""
import pytest
import json
from datasets import Dataset
from ragas import evaluate
# 修复 ragas 新版本导入错误
from eval_type_backport import eval_type_backport
from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall
import os
from rag_system import get_answer_and_context

# 加载测试集
with open("test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

QUESTIONS = test_data["questions"]
GROUND_TRUTHS = test_data["ground_truths"]



@pytest.fixture(scope="module")
def eval_dataset():
    """构建RAGAS评估所需的数据集"""
    answers = []
    contexts_list = []
    for q in QUESTIONS:
        ans, ctx = get_answer_and_context(q)
        answers.append(ans)
        contexts_list.append(ctx)
    return Dataset.from_dict({
        "question": QUESTIONS,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": GROUND_TRUTHS
    })

# ========== 测试用例 ==========
def test_faithfulness(eval_dataset):
    result = evaluate(dataset=eval_dataset, metrics=[faithfulness])
    score = result["faithfulness"]
    assert score > 0.7, f"忠实度太低: {score}"

def test_answer_relevancy(eval_dataset):
    result = evaluate(dataset=eval_dataset, metrics=[answer_relevancy])
    score = result["answer_relevancy"]
    assert score > 0.7, f"答案相关性太低: {score}"

def test_context_recall(eval_dataset):
    result = evaluate(dataset=eval_dataset, metrics=[context_recall])
    score = result["context_recall"]
    assert score > 0.6, f"上下文召回率太低: {score}"

# 生成详细报告
def test_generate_report(eval_dataset):
    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )
    df = result.to_pandas()
    df.to_csv("rag_eval_report.csv", index=False)
    print("\n评估报告已保存到 rag_eval_report.csv")
    print(df)
