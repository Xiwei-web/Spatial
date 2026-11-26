"""所有作业都经 scripts/submit.sh 提交，且一律带上 configs/slurm.yaml 里的排除节点。"""
import os
import re
import subprocess

import yaml

from src import paths

R = paths.REPO_ROOT


def test_no_direct_sbatch_or_srun_outside_submit_sh():
    bad = []
    for d in ("scripts", "src"):
        for root, _, files in os.walk(R / d):
            for f in files:
                p = os.path.join(root, f)
                if not f.endswith((".sh", ".py")) or os.path.samefile(p, R / "scripts" / "submit.sh"):
                    continue
                for i, line in enumerate(open(p, errors="ignore"), 1):
                    if re.search(r"\b(sbatch|srun)\b", line):
                        bad.append(f"{os.path.relpath(p, R)}:{i}: {line.strip()[:100]}")
    assert not bad, "除 submit.sh 外不得调用 sbatch / srun：\n" + "\n".join(bad)


def _dry(*args, cfg=None):
    env = {**os.environ, "SUBMIT_DRYRUN": "1"}
    if cfg is not None:
        env["INTERP_SLURM_FILE"] = str(cfg)
    return subprocess.run(["bash", str(R / "scripts" / "submit.sh"), *args], capture_output=True, text=True, env=env)


def test_exclude_matches_config_for_sbatch_and_srun():
    want = ",".join(yaml.safe_load(open(R / "configs" / "slurm.yaml"))["exclude_nodes"])
    assert want == "gpu-05,gpu-07,gpu-49"
    for args in (("gpu1", "--time=01:00:00", "--name=t", "echo", "hi"), ("--srun", "gpu2", "--name=t", "echo", "hi"), ("cpu", "echo", "hi")):
        r = _dry(*args)
        assert r.returncode == 0, r.stderr
        assert f"--exclude={want}" in r.stdout, r.stdout
    assert _dry("--srun", "gpu1", "echo").stdout.startswith("srun ")
    assert _dry("gpu1", "echo").stdout.startswith("sbatch ")


def test_missing_or_empty_config_fails(tmp_path):
    assert _dry("gpu1", "echo", cfg=tmp_path / "nope.yaml").returncode != 0
    (tmp_path / "s.yaml").write_text("exclude_nodes: []\n")
    assert _dry("gpu1", "echo", cfg=tmp_path / "s.yaml").returncode != 0
    (tmp_path / "t.yaml").write_text("exclude_nodes: [gpu-99]\n")
    assert "--exclude=gpu-99" in _dry("gpu1", "echo", cfg=tmp_path / "t.yaml").stdout
