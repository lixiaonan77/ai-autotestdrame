"""
多模态RAG测试：本地图片 + OCR + 图文检索
"""
import os
import pytest
import easyocr

IMAGE_DIR = "images"
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

TEST_CASES = [
    ("apple.jpg", "图片里的水果有什么营养？", "维生素C"),
    ("banana.jpg", "图片中的水果有什么特点？", "钾元素"),
    ("car.jpg", "这是什么的动力来源？", "汽油或电力"),
    (None, "苹果是什么颜色？", "红色"),
]

def ocr_image(path):
    if not path or not os.path.exists(path):
        return ""
    try:
        return " ".join(reader.readtext(path, detail=0))
    except:
        return ""

@pytest.mark.parametrize("img, question, keyword", TEST_CASES)
def test_multimodal(img, question, keyword):
    full_path = os.path.join(IMAGE_DIR, img) if img else None
    img_text = ocr_image(full_path)
    query = f"图片内容：{img_text} 问题：{question}"
    
    from rag_system import get_answer_and_context
    ans, _ = get_answer_and_context(query)
    
    assert keyword in ans
    print(f"✅ 多模态测试成功：{question} => {ans}")
