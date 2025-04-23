from langchain_ollama import OllamaLLM
from langchain.agents import AgentType, initialize_agent, load_tools

# 1. 初始化模型
llm = OllamaLLM(model="deepseek-r1:8b")

# 2. 加载工具（DuckDuckGo搜索）
tools = load_tools(["ddg-search"])  # 无需API Key

# 3. 创建Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # 显示详细执行过程
)

# 4. 执行联网查询
response = agent.run("铁道供电用英文,马来语怎么说")
print(response)