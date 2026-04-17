"""
test_rag_performance.py
RAG 性能测试：测量 检索耗时、生成耗时、总耗时
"""
import pytest

# 导入自己写的 RAG 函数
from rag_system import get_answer_and_context, compression_retriever, llm_client,MODEL_NAME

# 测试用的问题（固定5题）
PERF_QUESTIONS = [
    "苹果有什么营养价值？",
    "香蕉有哪些特点？",
    "汽车的动力来源是什么？",
    "苹果和香蕉都是水果吗？",
    "特斯拉续航多少？"
]

# ==========================
# 1. 测试 端到端延迟（检索+LLM生成）
# ==========================
@pytest.mark.benchmark(group="RAG端到端")
def test_e2e_latency(benchmark):
    """测量整个RAG流程耗时：检索 + 大模型回答"""
    
    # 定义要测速的函数
    def e2e(question):
        return get_answer_and_context(question)

    # pedantic：严谨模式，跑 10 轮，每轮跑 5 次
    result = benchmark.pedantic(
        e2e, 
        args=(PERF_QUESTIONS[0],),  # 测第一个问题
        iterations=5,   # 每轮跑5次
        rounds=10       # 跑10轮统计
    )

# ==========================
# 2. 仅测试检索（不调用大模型）
# ==========================
@pytest.mark.benchmark(group="检索单独")
def test_retrieval_latency(benchmark):
    """只测向量检索 + 重排序，不生成答案"""
    
    def retrieval_only(question):
        docs = compression_retriever.invoke(question)
        return [doc.page_content for doc in docs]

    benchmark.pedantic(retrieval_only, args=(PERF_QUESTIONS[0],), iterations=5, rounds=10)

# ==========================
# 3. 仅测试LLM生成（不检索）
# ==========================
@pytest.mark.benchmark(group="生成单独")
def test_generation_latency(benchmark):
    """只测大模型生成，不检索，固定上下文"""
    
    fixed_context = "苹果富含维生素C和膳食纤维。"

    def generation_only(question):
        prompt = f"参考资料：{fixed_context}\n问题：{question}\n答案："
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        return response.choices[0].message.content

    benchmark.pedantic(generation_only, args=("苹果有什么营养？",), iterations=5, rounds=10)
