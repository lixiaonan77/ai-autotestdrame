from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType,Tool
from day10_rages import   query_fruit_info
import os

fruit_tool=Tool(
    name="fruit",
    func=query_fruit_info,
    description="查询水果相关信息，例如苹果香蕉"
    )
# 工具
TOOLS_TO_LOAD = ["llm-math"]

# 问题
QUESTION = "100的平方根加上25的平方根等于多少？"

# 真正有用的大模型
llm = OpenAI(
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 加载工具
tools = load_tools(TOOLS_TO_LOAD, llm=llm)
tools.append(fruit_tool)

# 初始化智能体
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 运行
response = agent.run(QUESTION)
print(f"\n最终答案：{response}")
