"""
RAG系统质量自动化测试
使用RAGAS评估指标 + pytest
"""
import pytest
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
import os
from rag_system import compression_retriever, llm_client, MODEL_NAME, SYSTEM_ROLE

# 加载测试集
with open("test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

QUESTIONS = test_data["questions"]
GROUND_TRUTHS = test_data["ground_truths"]

def get_answer_and_context(question: str):
    """调用RAG系统，返回答案和检索到上下文列表"""
    # 1.检索
    docs = compression_retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    context_str = "\n".join(contexts)

    # 2.生成答案（带上角色）
    prompt = f"""
{SYSTEM_ROLE}

根据以下参考资料回答问题。如果参考资料中没有相关信息，就说“不知道”。
参考资料：
{context_str}
问题：{question}
答案：
"""
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    answer = response.choices[0].message.content
    return answer, contexts

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
