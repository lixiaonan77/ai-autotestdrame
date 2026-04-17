# run_all_tests.py
# 一键运行：质量 + 对抗 + 性能 + 鲁棒性 全部测试
import pytest
import sys

def run_all_tests():
    print("=== 开始运行 RAG+Agent 全量质量测试 ===")

    args = [
        "test_rag_quality.py",      # RAGAS 质量评估
        "test_adversarial.py",      # 对抗性测试
        "test_rag_performance.py",  # 性能测试
        "test_robustness.py",       # 鲁棒性测试
        "-v",
        "--html=combined_report.html",  # 输出综合报告
        "--self-contained-html"
    ]

    # 运行 pytest
    exit_code = pytest.main(args)
    print(f"测试结束，退出码：{exit_code}")

    # 非0 = 测试失败
    sys.exit(exit_code)

if __name__ == "__main__":
    run_all_tests()
