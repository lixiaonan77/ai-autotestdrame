# ==================== 1. 导入依赖 ====================
import os
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.llms import OpenAI

# ==================== 2. 设置 OpenAI API Key ====================
# 方式一：在终端执行 export OPENAI_API_KEY="sk-xxx"
# 方式二：直接在代码中设置（不推荐提交到公开仓库）
# os.environ["OPENAI_API_KEY"] = "your-key-here"

# ==================== 3. 定义私有数据查询函数（模拟版） ====================
def query_cat_info(question: str) -> str:
    """
    模拟一个 RAG 查询函数。
    实际使用中，你可以替换为你的真实 RAG 链（比如 qa_chain.run）。
    """
    # 模拟的私有知识库内容
    cat_knowledge = {
        "喜欢吃什么": "三文鱼",
        "什么时候等零食": "每天下午3点",
        "名字": "Whiskers",
        "颜色": "橘色",
        "性格": "粘人，喜欢晒太阳"
    }
    # 简单的关键词匹配（真实场景应使用 RAG 检索）
    for key, value in cat_knowledge.items():
        if key in question:
            return value
    return "我暂时不知道关于这个问题的答案，请换个问法。"

# 如果你想使用真实的 RAG 系统（来自 Day10），请取消注释下面的代码：
from your_rag_code import qa_chain   # 替换为你的 RAG 脚本名
def query_cat_info(question: str) -> str:
     return qa_chain.run(question)

# ==================== 4. 将函数封装成 LangChain 工具 ====================
cat_tool = Tool(
    name="CatInfo",
    func=query_cat_info,
    description="查询关于猫 Whiskers 的信息。输入应该是一个具体问题，例如：'猫喜欢吃什么？'"
)

# ==================== 5. 准备其他工具（例如计算器） ====================
from langchain.agents import load_tools
llm = OpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)   # 加载计算器工具
tools.append(cat_tool)                      # 添加猫信息工具

# ==================== 6. 初始化 Agent ====================
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True   # 打印思考过程
)

# ==================== 7. 测试 Agent ====================
print("=" * 50)
print("测试1：单独问猫的信息")
response1 = agent.run("Whiskers 喜欢吃什么？")
print(f"最终答案：{response1}\n")

print("=" * 50)
print("测试2：结合计算器的问题（需要多步推理）")
response2 = agent.run("Whiskers 每天吃三文鱼两次，一周总共吃几次？")
print(f"最终答案：{response2}\n")

print("=" * 50)
print("测试3：模糊问题（Agent 会尝试推理）")
response3 = agent.run("我的猫什么时候会蹲在冰箱前？")
print(f"最终答案：{response3}")
