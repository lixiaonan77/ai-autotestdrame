"""
test_adversarial.py
对抗性测试：检验 RAG 系统在边界、否定、跨文档、数值、模糊指代等场景下的表现
"""

# 导入 pytest 自动化测试框架
import pytest
# 从 test_rag_quality 文件中导入 RAG 问答函数：输入问题 → 返回答案 + 检索上下文
from rag_system import get_answer_and_context

# ==================== 校验函数 ====================
# 这些函数用来判断 RAG 回答是否符合要求

def check_unknown(answer: str) -> bool:
    """
    功能：检查答案是否表达了'不知道'或类似含义
    返回：只要出现任意一个关键词 → True
    """
    # 定义“不知道”相关关键词列表
    unknown_keywords = ["不知道", "没有找到", "无法回答", "没有相关信息", "暂未找到", "未提及"]
    # any()：只要有一个关键词在答案里 → 返回 True
    return any(kw in answer for kw in unknown_keywords)

def check_affirmative(answer: str) -> bool:
    """
    功能：检查答案是否表达了肯定态度（是、是的、对）
    """
    affirmative_keywords = ["是", "是的", "对", "没错", "属于", "都是", "均属于"]
    return any(kw in answer for kw in affirmative_keywords)

def check_negative_rebuttal(answer: str) -> bool:
    """
    功能：针对否定问题（苹果一定不是甜的吗？），应正确反驳
    检查是否出现：不对、不一定、错误 等
    """
    rebuttal_keywords = ["不对", "不一定", "有甜", "也有不甜的", "错误", "并非"]
    return any(kw in answer for kw in rebuttal_keywords)

def check_number(answer: str, expected: str) -> bool:
    """
    功能：检查答案中是否包含预期数字（如 4、C）
    """
    return expected in answer

def check_inference(answer: str, expected_fruit: str) -> bool:
    """
    功能：检查模糊指代问题是否推理出正确对象（红色水果 → 苹果）
    """
    return expected_fruit in answer

def check_contains_all(answer: str, keywords: list) -> bool:
    """
    功能：检查答案是否包含所有给定关键词
    all() → 必须全部满足才返回 True
    """
    return all(kw in answer for kw in keywords)

# ==================== 测试用例定义 ====================
# 格式：(问题, 校验函数, 预期值或None, 失败时提示信息)
# 每一行 = 一道对抗测试题
test_cases = [
    # 1. 知识库外 → 必须拒答
    ("西瓜是甜的么？", check_unknown, None, "知识库外问题未正确拒答"),
    ("特斯拉的续航里程是多少？", check_unknown, None, "知识库外问题未正确拒答"),
    ("榴莲有什么功效？", check_unknown, None, "知识库外问题未正确拒答"),

    # 2. 跨文档推理 → 需要综合多条信息
    ("苹果和香蕉都是日常水果吗？", check_affirmative, None, "跨文档推理失败，未给出肯定答案"),
    ("苹果和汽车哪个是交通工具？", check_affirmative, None, "跨文档推理失败，应指出汽车是交通工具"),

    # 3. 否定词问题 → 不能被带偏，要正确反驳
    ("苹果一定不是甜的，对吗？", check_negative_rebuttal, None, "否定问题未正确反驳"),
    ("香蕉不富含钾元素，是吗？", check_negative_rebuttal, None, "否定问题未正确反驳"),

    # 4. 数值/细节精确性 → 不能瞎编
    ("汽车有几个轮子？", check_number, "4", "轮子数量错误"),
    ("苹果含有哪种维生素？", check_number, "C", "维生素信息错误或缺失"),
    ("香蕉富含什么元素？", check_contains_all, ["钾"], "香蕉应提到钾元素"),

    # 5. 模糊指代 → 能推理出正确答案
    ("那个红色的水果有什么好处？", check_inference, "苹果", "未能从'红色水果'推理出苹果"),
]

# ==================== 参数化测试 ====================
# @pytest.mark.parametrize：自动循环跑所有测试用例
# 自动把 test_cases 里的每一组数据拆成：question, check_func, expected, msg
@pytest.mark.parametrize("question,check_func,expected,msg", test_cases)
def test_adversarial(question, check_func, expected, msg):
    """
    对抗性测试主函数
    1. 输入问题
    2. 获取 RAG 答案
    3. 根据校验函数判断是否正确
    """
    # 调用 RAG 系统，得到答案 + 检索到的上下文
    answer, contexts = get_answer_and_context(question)
    
    # 如果没有预期值（如 check_unknown / check_affirmative）
    if expected is None:
        # 执行校验函数，判断是否通过
        assert check_func(answer), f"{msg}\n问题: {question}\n答案: {answer}\n检索到的上下文: {contexts}"
    else:
        # 根据不同校验函数，执行不同验证逻辑
        if check_func == check_number:
            # 验证数字
            assert check_func(answer, expected), f"{msg}\n问题: {question}\n答案: {answer}\n预期包含数字: {expected}"
        elif check_func == check_inference:
            # 验证模糊推理
            assert check_func(answer, expected), f"{msg}\n问题: {question}\n答案: {answer}\n预期推理出: {expected}"
        elif check_func == check_contains_all:
            # 验证是否包含所有关键词
            assert check_func(answer, expected), f"{msg}\n问题: {question}\n答案: {answer}\n预期包含所有关键词: {expected}"
        else:
            # 其他通用校验
            assert check_func(answer), f"{msg}\n问题: {question}\n答案: {answer}"

# ==================== 运行测试并生成报告 ====================
if __name__ == "__main__":
    # 运行测试，输出详细日志 + 生成 HTML 测试报告
    pytest.main([__file__, "-v", "--html=adversarial_report.html", "--self-contained-html"])
