"""
AI 全链路质量保障一体化测试
包含：鲁棒性、RAG质量、RAG性能、对抗、红队、Agent记忆、多模态本地图片
全部对接真实 API 版本 rag_system.py
"""
import subprocess
import sys

if __name__ == "__main__":
    print("=== 开始 AI 全链路自动化测试 ===")

    subprocess.run([
        sys.executable, "-m", "pytest",
        # 基础测试
        "test_robustness.py",
        "test_adversarial.py",
        
        # RAG 质量+性能（API 版）
        "test_rag_quality.py",
        "test_rag_performance.py",
        
        # 红队安全测试
        "test_redteam.py",
        
        # Agent 记忆测试
        "test_agent_memory.py",
        
        # 多模态：本地批量图片 + 真实 OCR
        "test_multimodal_local_images.py",

        # 输出格式
        "-v",
        "--html=ai_full_link_test_report.html"
    ])

    print("\n✅ 全部测试完成！报告：ai_full_link_test_report.html")
