# 使用 RAGAS 框架评估 RAG 系统性能

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from openai import OpenAI
import os

# ===================== 角色设定（你要的 service team 风格）=====================
SERVICE_TEAM = "智能测试工程师"
SYSTEM_ROLE = f"""你是一名专业的{SERVICE_TEAM}，回答严格依据参考资料，简洁、准确、严谨，不编造内容。"""

# 创建客户端
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
MODEL_NAME = "deepseek-chat"

# 向量模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

doc_content = '''
苹果是一种水果，通常有红色、绿色或黄色。苹果富含维生素C和膳食纤维。
香蕉也是一种水果，表皮黄色，果肉软甜，富含钾元素。
汽车是一种交通工具，使用汽油或电力驱动，有四个轮子。
'''

DOC_PATH = "test_doc.txt"
if not os.path.exists(DOC_PATH):
    with open("test_doc.txt", "w", encoding="utf-8") as f:
        f.write(doc_content)

# 加载并切分文档
loader = TextLoader("test_doc.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# 构建向量库
vectorstore = FAISS.from_documents(docs, embedding_model)

# 带重排序的检索器
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

def query_fruit_info(question: str) -> str:
    """单个问题查询函数，供 Agent 调用"""
    retrieved_docs = compression_retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]
    context_str = "\n".join(contexts)
    
    prompt = f"""
{SYSTEM_ROLE}

请根据以下参考资料回答用户的问题。如果参考资料中没有相关信息，就说“不知道”。

参考资料：
{context_str}

用户问题：{question}
答案："""
    
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )
    return response.choices[0].message.content

# --- 运行测试 ---
if __name__ == "__main__":
    test_questions = [
        "苹果有什么营养价值？",
        "香蕉有哪些特点？",
        "汽车的动力来源是什么？"
    ]
    for q in test_questions:
        print("=" * 50)
        print(f"问题：{q}")
        result = query_fruit_info(q)
        print(f"回答：{result}")
