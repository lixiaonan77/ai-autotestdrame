# 使用 RAGAS 框架评估 RAG 系统性能（RAGAS：用于评估检索增强生成系统的专业框架）
from langchain_community.document_loaders import TextLoader  # 文档加载器：加载文本文件
from langchain_text_splitters import RecursiveCharacterTextSplitter
 # 文本分割器：递归按字符分割文本
from langchain_community.vectorstores import FAISS  # 向量数据库：用于存储文档向量，支持快速检索
from langchain_huggingface import HuggingFaceEmbeddings  # 嵌入模型：用于将文本转换为向量
from langchain.retrievers import ContextualCompressionRetriever  # 上下文压缩检索器：优化检索结果
from langchain.retrievers.document_compressors import CrossEncoderReranker  # 重排序器：对检索结果进行二次排序
from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # 交叉编码器：用于重排序的模型
from openai import OpenAI  # OpenAI 客户端：用于调用大模型（此处适配 DeepSeek API）
import os  # 系统模块：用于处理文件路径、环境变量
import easyocr
import pytest

# ===================== 角色设定（你要求的 service team 风格）=====================
SERVICE_TEAM = "智能测试工程师"  # 角色名称：定义系统扮演的角色
SYSTEM_ROLE = f"""你是一名专业的{SERVICE_TEAM}，回答严格依据参考资料，简洁、准确、严谨，不编造内容。"""  # 角色提示词：约束大模型回答规范
'''
# 创建大模型客户端（适配 DeepSeek API，重点报错处理）
# 报错说明：若出现"link dead"（链接失效）或"link hit security strategy"（触发安全策略），按下方备注排查
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量中获取 DeepSeek API 密钥（避免硬编码泄露）
    base_url="https://api.deepseek.com"  # DeepSeek API 基础地址（报错需优先检查该地址是否可正常访问）
)
MODEL_NAME = "deepseek-chat"  # 使用的 DeepSeek 模型名称（固定为deepseek-chat，无需修改）
'''
# 新增：豆包API客户端（免费可用，适配OpenAI格式）
llm_client = OpenAI(
    api_key=os.getenv("DOUBAO_API_KEY"),  # 从环境变量获取豆包API密钥
    base_url="https://ark.cn-beijing.volces.com/api/v3" # 豆包API基础地址
)
MODEL_NAME = "doubao-seed-1-6-251015"  # 豆包免费模型名称
# 向量嵌入模型（中文适配，轻量高效）
# 模型说明：BAAI/bge-small-zh-v1.5 是轻量级中文嵌入模型，适配中文文档检索，无需额外配置
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 测试文档内容（用于构建RAG系统的知识库，可根据需求修改）
doc_content = '''
苹果是一种水果，通常有红色、绿色或黄色。苹果富含维生素C和膳食纤维。
香蕉也是一种水果，表皮黄色，果肉软甜，富含钾元素。
汽车是一种交通工具，使用汽油或电力驱动，有四个轮子。
'''

# 文档保存路径（本地生成测试文档，路径可自定义）
DOC_PATH = "test_doc.txt"
# 若文档不存在，则自动创建并写入上述知识库内容（避免手动创建文档）
if not os.path.exists(DOC_PATH):
    with open("test_doc.txt", "w", encoding="utf-8") as f:
        f.write(doc_content)

# 全局变量：用于存储向量库和检索器，供重载函数调用（解决reload函数未定义问题）
global vectorstore, compression_retriever

# 加载并切分文档（构建知识库的核心步骤，复用逻辑，避免代码冗余）
def load_documents_and_build_vectorstore():
    """加载文档、切分文档并构建向量库，供初始化和知识库重载使用
    功能：读取本地文档→分割为小块→转换为向量→构建检索器，是RAG系统的基础
    """
    global vectorstore, compression_retriever
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    # 加载本地测试文档（指定编码为utf-8，避免中文乱码）
    loader = TextLoader("test_doc.txt", encoding="utf-8")
    documents = loader.load()  # 读取文档内容，返回文档对象列表
    # 初始化文本分割器：chunk_size=100（每块最大100字符），chunk_overlap=20（块之间重叠20字符，保证上下文连贯）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)  # 分割文档为多个小块，适配向量库存储
    # 构建FAISS向量库（将分割后的文档块转换为向量并存储，支持快速检索）
    vectorstore = FAISS.from_documents(docs, embedding_model)
    # 构建带重排序的检索器（提升检索准确性：先检索，再二次排序，过滤无关文档）
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # 基础检索器：检索前10个相关文档
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")  # 重排序模型：优化检索结果排序
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)  # 压缩器：保留排序后前3个最相关文档
    # 上下文压缩检索器（整合基础检索器和重排序器，最终用于检索相关文档）
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

# 初始化知识库（程序启动时自动执行，构建初始向量库和检索器）
load_documents_and_build_vectorstore()

# 新增：实现reload_knowledge_base()函数（解决原代码调用未定义函数的问题）
def reload_knowledge_base():
    """知识库重载函数（核心功能：更新知识库后，重新加载文档并重建向量库）
    适用场景：当test_doc.txt文档内容修改后，调用该函数即可加载新的知识库
    运行提示：重载成功后，终端会打印提示信息，确认重载生效
    """
    # 1. 重新读取本地文档内容（模拟知识库更新，读取修改后的test_doc.txt）
    with open("test_doc.txt", "r", encoding="utf-8") as f:
        global doc_content
        doc_content = f.read()
    # 2. 重新加载文档并构建向量库、检索器（复用初始化逻辑，避免代码冗余）
    load_documents_and_build_vectorstore()
    print("知识库已成功重载，最新文档内容已生效！")

def query_fruit_info(question: str) -> str:
    """单个问题查询函数，供 Agent 调用
    参数：question - 用户提问（字符串），支持各类正常/异常输入（适配鲁棒性测试）
    返回：大模型生成的回答（字符串），严格遵循参考资料，无相关信息则返回"不知道"
    """
    retrieved_docs = compression_retriever.invoke(question)  # 检索与问题相关的文档（调用重排序检索器）
    contexts = [doc.page_content for doc in retrieved_docs]  # 提取检索到的文档内容，用于生成回答
    context_str = "\n".join(contexts)  # 将多个文档内容拼接为字符串，作为大模型参考资料
    
    # 构建提示词（带上角色设定，确保大模型回答符合要求，不编造、不误导）
    prompt = f"""
{SYSTEM_ROLE}

请根据以下参考资料回答用户的问题。如果参考资料中没有相关信息，就说“不知道”。

参考资料：
{context_str}

用户问题：{question}
答案："""
    
    # 调用 DeepSeek 大模型生成回答（适配OpenAI客户端格式，参数固定，无需修改）
    response = llm_client.chat.completions.create(
        model="doubao-seed-1-6-251015",  # 使用的模型，与上方MODEL_NAME一致
        messages=[{"role": "user", "content": prompt}],  # 对话消息（用户角色+提示词+参考资料+问题）
        temperature=0,  # 温度：0表示回答更严谨、无随机波动，避免大模型编造内容
        max_tokens=300,  # 最大生成 tokens 数：控制回答长度，防止回答过长
    )
    return response.choices[0].message.content  # 返回生成的回答内容（提取大模型响应结果）

def get_answer_and_context(question: str):
    """调用RAG系统，返回答案和检索到的上下文列表（适配鲁棒性测试脚本）
    参数：question - 用户提问（字符串），支持各类异常输入测试
    返回：answer（回答内容，字符串）、contexts（检索到的上下文列表，供测试验证）
    """
    # 1. 检索：获取与问题相关的文档（调用压缩检索器，获取最相关的3个文档）
    docs = compression_retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]  # 提取上下文内容，用于测试时验证检索准确性
    context_str = "\n".join(contexts)  # 拼接上下文，作为大模型参考资料
    
    # 2. 生成答案（带上角色设定，确保回答严谨、不编造，符合鲁棒性测试要求）
    prompt = f"""
{SYSTEM_ROLE}

根据以下参考资料回答问题。如果参考资料中没有相关信息，就说“不知道”。
参考资料：
{context_str}
问题：{question}
答案：
"""
    # 调用 DeepSeek 大模型生成回答（参数与query_fruit_info函数一致，保证回答规范）
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    answer = response.choices[0].message.content  # 提取回答内容
    return answer, contexts  # 返回回答和上下文列表（适配test_robustness.py测试脚本）
# ==========================
# 【新增：多模态RAG函数（本地图片+文字）】
# ==========================
def ocr_image(image_path):
    if not image_path or not os.path.exists(image_path):
        return ""
    try:
        result = reader.readtext(image_path, detail=0)
        return " ".join(result)
    except:
        return ""

def multimodal_rag_query(image_path, question):
    """
    多模态查询：本地图片 + 文字
    供 test_multimodal_local_images.py 调用
    """
    image_text = ocr_image(image_path)
    full_query = f"图片内容：{image_text} 用户问题：{question}"
    return get_answer_and_context(full_query)

# ==========================
# 【新增：Agent多轮记忆函数】
# ==========================
def agent_multi_turn_query(history: list, question: str):
    """
    多轮对话记忆测试
    history: 历史对话列表
    question: 当前问题
    """
    context = "\n".join([f"Q:{q}\nA:{a}" for q,a in history])
    prompt = f"历史对话：{context}\n当前问题：{question}"
    ans, ctx = get_answer_and_context(prompt)
    return ans, ctx
# --- 运行测试（验证当前RAG系统是否正常工作，含知识库重载调用）---
if __name__ == "__main__":
    # 调用知识库重载函数（验证重载功能，可注释该句，测试初始知识库）
    reload_knowledge_base()
    # 测试问题列表（覆盖知识库中的苹果、香蕉、汽车相关内容，验证系统响应）
    test_questions = [
        "苹果有什么营养价值？",
        "香蕉有哪些特点？",
        "汽车的动力来源是什么？"
    ]
    # 循环执行测试，打印结果（直观查看系统回答是否符合预期）
    for q in test_questions:
        print("=" * 50)  # 分隔线：区分不同问题的测试结果
        print(f"问题：{q}")
        result = query_fruit_info(q)  # 调用查询函数，获取回答
        print(f"回答：{result}")
