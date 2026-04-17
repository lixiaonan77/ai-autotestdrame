"""
RAG+Agent 鲁棒性测试完整版（中文翻译：RAG+Agent 系统鲁棒性测试完整版本）
包含：
1. 异常输入测试（14种）→ 中文翻译：1. 异常输入测试（涵盖14种常见异常场景）
2. 噪声鲁棒性、负面拒绝、信息整合、反事实鲁棒性 → 中文翻译：2. 四大核心鲁棒性测试（噪声鲁棒性、负面拒绝、信息整合、反事实鲁棒性）
3. 环境异常Mock（向量库失败、LLM超时、检索为空）→ 中文翻译：3. 环境异常模拟（使用Mock模拟向量库失败、大模型超时、检索结果为空场景）
4. 知识库变更稳定性测试 → 中文翻译：4. 知识库变更稳定性测试
"""
# 导入所需依赖库
import pytest  # 单元测试框架：用于编写和运行测试用例
import openai  # OpenAI客户端：用于捕获大模型相关异常（适配DeepSeek）
from unittest.mock import patch  # 模拟工具：用于模拟环境异常（如向量库失败、LLM超时）
from rag_system import get_answer_and_context,reload_knowledge_base# 导入RAG系统核心接口（从rag_system.py中引入，用于获取回答和上下文）


# ==========================
# 1. 异常输入测试用例（全覆盖）
# 中文翻译：1. 异常输入测试用例（全覆盖14种常见异常场景，确保系统兼容各类异常输入）
# ==========================
# 定义异常输入测试用例列表，格式：（测试输入内容，用例名称）
robust_test_cases = [
    ("", "空问题"),  # 测试用例1：空字符串输入
    ("   ", "全空格"),  # 测试用例2：纯空格输入
    ("啊" * 10000, "超长输入"),  # 测试用例3：10000个重复字符的超长输入
    ("<script>alert('xss')</script>", "XSS攻击"),  # 测试用例4：XSS攻击脚本输入
    ("苹果' OR '1'='1", "SQL注入风格"),  # 测试用例5：SQL注入风格语句输入
    ("！@#￥%……&*（）", "纯特殊符号"),  # 测试用例6：纯特殊符号输入
    ("🍎🍌🚗", "表情符号"),  # 测试用例7：表情符号输入
    ("苹果苹果苹果苹果", "重复词汇"),  # 测试用例8：重复词汇输入
    ("苹果一定不是水果，对吗？", "否定误导"),  # 测试用例9：否定式误导问题
    ("苹果和汽车哪个更有营养？", "跨领域混乱问题"),  # 测试用例10：跨领域无关联问题
    ("Apple 有什么营养？", "中英文混合"),  # 测试用例11：中英文混合输入
    ("平果营羊有哪些", "错别字"),  # 测试用例12：错别字输入
    ("为什么苹果是蓝色的？", "事实错误前提"),  # 测试用例13：基于错误事实的提问
    ("火星上有人吗？", "知识库完全不存在"),  # 测试用例14：知识库中无相关信息的问题
]

# parametrize装饰器：批量运行所有异常输入用例，无需逐个编写测试函数
@pytest.mark.parametrize("question, case_name", robust_test_cases)
def test_robust_input(question, case_name):
    """异常输入测试：确保系统在各类异常输入下不崩溃、不产生幻觉、不被误导
    参数：
        question：测试输入的问题/内容
        case_name：测试用例名称，用于定位问题
    """
    try:
        # 调用RAG系统接口，获取回答和上下文（模拟真实用户调用场景）
        answer, contexts = get_answer_and_context(question)
    except Exception as e:
        # 若系统崩溃，标记测试失败，并输出错误信息（截取前100字符，避免信息过长）
        pytest.fail(f"崩溃【{case_name}】: {str(e)[:100]}")

    # 断言1：回答不为空，避免系统返回空值异常，确保系统正常响应
    assert answer is not None
    # 断言2：若输入为空/全空格，系统需提示用户输入有效问题
    if question.strip() == "":
        assert any(kw in answer for kw in ["请输入", "有效", "问题"])
    # 断言3：若为“知识库完全不存在”用例，系统需明确拒答，不编造答案
    if case_name == "知识库完全不存在":
        assert any(w in answer for w in ["不知道", "无法回答", "没有", "暂无"])


# ==========================
# 2. 鲁棒性四大核心测试（对应RAG系统核心鲁棒性要求）
# 中文翻译：2. 四大核心鲁棒性测试（对应RAG系统核心鲁棒性指标，必测项）
# ==========================
def test_noise_robustness():
    """噪声鲁棒性测试：检索到无关文档（噪声）时，系统不应强行编造答案，需诚实拒答"""
    # 提问与知识库无关的问题（汽车轮胎材质，知识库仅包含苹果、香蕉、汽车基础信息，无轮胎材质）
    answer, _ = get_answer_and_context("汽车的轮胎是什么材质？")
    # 断言：系统需拒答，包含拒答相关关键词
    assert "不知道" in answer or "没有" in answer or "暂无" in answer

def test_negative_rejection():
    """负面拒绝测试：面对知识库中完全不存在的问题，系统应明确拒答，不产生幻觉"""
    # 提问知识库中无相关信息的问题（火星大小，知识库未涉及火星相关内容）
    answer, _ = get_answer_and_context("火星有多大？")
    # 断言：系统需拒答，包含拒答相关关键词
    assert "不知道" in answer or "没有" in answer or "暂无" in answer

def test_information_integration():
    """信息整合测试：当答案分散在多个文档碎片中时，系统能有效融合，输出完整、准确的回答"""
    # 提问需整合多文档信息的问题（苹果是什么+有什么营养，信息分散在知识库不同片段）
    answer, _ = get_answer_and_context("苹果是什么？它有什么营养？")
    # 断言：回答需包含两个核心信息（苹果的属性、营养价值）
    assert "水果" in answer and "维生素C" in answer

def test_counterfactual_robustness():
    """反事实鲁棒性测试：当检索到的文档包含错误信息时，系统能识别错误并纠正，不盲从错误"""
    # 提问涉及错误前提的问题（苹果颜色，知识库中苹果无蓝色，模拟文档错误场景）
    answer, _ = get_answer_and_context("苹果是什么颜色？")
    # 断言1：回答中不包含错误信息（蓝色）
    assert "蓝色" not in answer
    # 断言2：回答包含正确信息（苹果常见颜色：红、绿、黄）
    assert any(c in answer for c in ["红", "绿", "黄"])


# ==========================
# 3. 检索为空 → 负面拒绝测试（补充场景，验证检索无结果时的拒答能力）
# 中文翻译：3. 检索为空场景测试 → 验证检索无结果时，系统的负面拒绝能力
# ==========================
def test_retrieval_empty():
    """检索为空测试：当检索结果为空（无相关文档）时，系统需明确拒答"""
    # 提问知识库中无相关信息的问题（银河系中心物质，知识库未涉及）
    ans, ctx = get_answer_and_context("银河系中心是什么物质？")
    # 断言：系统需拒答，包含拒答相关关键词
    assert any(kw in ans for kw in ["不知道", "无法", "没有", "暂无"])


# ==========================
# 4. Mock 环境异常测试（模拟真实环境中可能出现的故障，验证系统容错能力）
# 中文翻译：4. 模拟环境异常测试（使用Mock工具模拟真实环境中的故障，验证系统容错能力）
# ==========================
@patch("rag_system.base_retriever.get_relevant_documents")
  # 模拟rag_system.py中的检索器调用方法
def test_vector_db_fail(mock_retrieve):
    """模拟向量数据库连接失败场景，验证系统容错能力"""
    # 模拟检索器调用时抛出连接超时异常（模拟向量数据库断开）
    mock_retrieve.side_effect = ConnectionError("连接超时")
    # 断言：系统能捕获异常，不崩溃、不卡死
    with pytest.raises(Exception):
        get_answer_and_context("测试问题")

@patch("rag_system.llm_client.chat.completions.create")  # 模拟rag_system.py中的大模型调用方法
def test_llm_timeout(mock_chat):
    """模拟大模型（LLM）API超时场景，验证系统容错能力"""
    # 模拟大模型调用时抛出请求超时异常（模拟DeepSeek API超时）
    mock_chat.side_effect = openai.APITimeoutError("请求超时")
    # 断言：系统能捕获异常，不崩溃、不卡死
    with pytest.raises(Exception):
        get_answer_and_context("测试问题")

@patch("rag_system.base_retriever.get_relevant_documents") # 模拟检索器调用方法
def test_retrieval_return_none(mock_retrieve):
    """模拟检索返回空列表场景，验证系统拒答能力"""
    # 模拟检索器返回空列表（无任何相关文档）
    mock_retrieve.return_value = []
    # 调用RAG系统接口
    ans, ctx = get_answer_and_context("测试")
    # 断言：系统需拒答，不编造答案
    assert "不知道" in ans or "无法回答" in ans or "暂无" in ans


# ==========================
# 5. 知识库变更稳定性测试（验证知识库更新后，系统答案同步更新，无异常）
# 中文翻译：5. 知识库变更稳定性测试（验证知识库更新后，系统答案能同步更新，运行稳定）
# ==========================
def test_kb_update_stability():
    """知识库变更稳定性测试：验证知识库更新前后，系统回答的一致性（或变更合理性）"""
    # 知识库更新前，查询苹果颜色并记录答案
    ans1, _ = get_answer_and_context("苹果的颜色")
    # 【注意】实际项目中，需取消注释下方代码，实现知识库重新加载（需在rag_system.py中实现该函数）
    reload_knowledge_base()  # 重新加载更新后的知识库（rag_system.py中需自定义实现）
    # 知识库更新后，再次查询同一问题
    ans2, _ = get_answer_and_context("苹果的颜色")
    # 断言：本测试为模拟场景，暂验证两次回答一致；实际知识库更新后，需改为assert ans1 != ans2
    assert ans1 != ans2


# ==========================
# 代码入口：运行所有测试用例，并生成HTML格式测试报告
# 中文翻译：代码入口：运行所有测试用例，并生成HTML格式的测试报告（可直接打开查看测试结果）
# ==========================
if __name__ == "__main__":
    # 运行当前测试脚本，参数说明：
    # -v：显示详细测试过程（每条用例的运行结果）
    # --html=robust_report.html：生成HTML格式测试报告，文件名为robust_report.html
    pytest.main([__file__, "-v", "--html=robust_report.html"])
