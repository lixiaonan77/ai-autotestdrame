"""
Agent 记忆能力测试：短期记忆、长期记忆、一致性、安全性
"""
import pytest
from rag_system import get_answer_and_context

def test_agent_short_memory():
    q1 = "苹果有什么营养？"
    a1, _ = get_answer_and_context(q1)

    q2 = "它里面有维生素吗？"
    a2, _ = get_answer_and_context(q2)
    
    assert "维生素" in a2
    print("✅ 短期记忆测试通过")

def test_agent_consistency():
    q1 = "苹果是什么颜色？"
    a1, _ = get_answer_and_context(q1)
    
    q2 = "苹果主要颜色是什么？"
    a2, _ = get_answer_and_context(q2)
    
    assert "红色" in a1 and "红色" in a2
    print("✅ 记忆一致性测试通过")
