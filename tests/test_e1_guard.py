"""E1 的 test 安全阀：数据里有 test 记录而 git HEAD 里没有 docs/PREREGISTRATION.md → 报错退出。"""
import subprocess

import pytest

from src import e1


class _P:
    def __init__(self, rc):
        self.returncode = rc


def _runner(rc):
    calls = []

    def run(cmd, capture_output=True):
        calls.append(cmd)
        return _P(rc)
    return run, calls


def test_dev_only_passes_without_prereg():
    run, calls = _runner(1)
    e1.require_prereg_if_test([{"split": "dev"}, {"stage": "screen"}], runner=run)
    assert calls == []          # 没有 test 记录就不查 git


@pytest.mark.parametrize("rec", [{"split": "test"}, {"side": "test"}, {"stage": "test"}])
def test_test_records_require_prereg(rec):
    run, calls = _runner(1)
    with pytest.raises(SystemExit):
        e1.require_prereg_if_test([{"split": "dev"}, rec], runner=run)
    assert calls and calls[0][-1] == "HEAD:docs/PREREGISTRATION.md"


def test_test_records_ok_with_prereg():
    run, _ = _runner(0)
    e1.require_prereg_if_test([{"split": "test"}], runner=run)


def test_real_repo_has_no_prereg_yet():
    """当前仓库 HEAD 里还没有预注册：真实的 git 调用应拒绝 test 记录（预注册提交后这个测试要删掉或改写）。"""
    if e1.prereg_committed():
        pytest.skip("预注册已提交")
    with pytest.raises(SystemExit):
        e1.require_prereg_if_test([{"split": "test"}])


def test_analysis_script_calls_guard():
    src = open("scripts/e1_analysis.py").read()
    i, j = src.index("require_prereg_if_test("), src.index("build_items(")
    assert i < j, "入口脚本必须在整理条目之前调用安全阀"
