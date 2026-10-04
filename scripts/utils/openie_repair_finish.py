"""Background completion gates for the isolated OpenIE index repair."""

import argparse
import fcntl
import logging
import os
import subprocess
import time
from pathlib import Path

from .openie_repair_smoke import ASSETS, ROOT, OUT_DEFAULT, read_json, sha256, write_json

LOG = logging.getLogger("openie.repair.finish")
PROPOSAL = ROOT / "docs/OpenIE_索引质量问题诊断与修复方案_2026-10-03.md"
RAG_PYTHON = "/root/anaconda3/envs/rag/bin/python"


def process_identity(pid):
    """Read the process start tick, so a reused PID cannot extend the wait."""
    directory = Path("/proc") / str(pid)
    try:
        stat = (directory / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        command = [item.decode() for item in (directory / "cmdline").read_bytes().split(b"\0") if item]
        if fields[0] == "Z" or not command:
            return None
        return {"pid": pid, "start_tick": int(fields[19]), "command": command}
    except FileNotFoundError:
        return None


def wait_for_repair(pid, out, poll_seconds=2):
    identity = process_identity(pid)
    if identity is None:
        raise ValueError(f"Repair PID {pid} is already absent; its completion cannot be inferred")
    scripts = {str(ROOT / "scripts/run_openie_quality_repair.sh"),
               str(ROOT / "scripts/repair_openie_index.py")}
    if not scripts.intersection(identity["command"]):
        raise ValueError(f"PID {pid} does not belong to this project's repair runner")
    command = identity["command"]
    if "--out-root" in command:
        supplied = Path(command[command.index("--out-root") + 1]).resolve()
    else:
        environment = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        override = next((item.split(b"=", 1)[1].decode() for item in environment
                         if item.startswith(b"OUT_ROOT=")), str(OUT_DEFAULT))
        supplied = Path(override).resolve()
    if supplied != out:
        raise ValueError("The waited repair uses a different output directory")
    with Path("/proc/stat").open() as stream:
        boot_epoch = next(int(line.split()[1]) for line in stream if line.startswith("btime "))
    identity["started_epoch"] = boot_epoch + identity["start_tick"] / os.sysconf("SC_CLK_TCK")
    LOG.info("[wait] existing repair PID=%s; no repair or model process is started", pid)
    while True:
        current = process_identity(pid)
        if current is None or current["start_tick"] != identity["start_tick"]:
            break
        time.sleep(poll_seconds)
    identity["wait_complete"] = True
    identity["exit_code"] = None  # This is not our child; validated artifacts establish success.
    return identity


def hashes_at(root, model_dir):
    return {name: sha256(root / model_dir / name) for name in ASSETS}


def require_validation(out, hippo_root, waited=None):
    validation_path = out / "validation_report.json"
    validation = read_json(validation_path)
    index = out / "repaired_index"
    if (validation.get("complete") is not True
            or validation.get("source_index_unchanged") is not True
            or validation.get("baseline_code_unchanged") is not True
            or validation.get("repaired_index") != str(index)
            or validation.get("invalid_records") != 0
            or validation.get("empty_chunks") != 0):
        raise ValueError("The repaired index has not passed all required validation gates")
    if waited and validation_path.stat().st_mtime < waited["started_epoch"]:
        raise ValueError("Validation predates the waited repair; refusing a stale success report")
    snapshot = read_json(out / "source_snapshot.json")
    model_dir = snapshot["model_dir"]
    if hashes_at(index, model_dir) != validation.get("asset_sha256"):
        raise ValueError("The repaired assets changed after validation")
    if hashes_at(Path(snapshot["source_index"]), model_dir) != snapshot["asset_sha256"]:
        raise ValueError("The frozen source index changed")
    tracked = subprocess.run(["git", "-C", str(hippo_root), "ls-files", "-z", "--", "*.py"],
                             check=True, capture_output=True).stdout
    baseline = {name.decode(): sha256(hippo_root / name.decode())
                for name in tracked.split(b"\0") if name and (hippo_root / name.decode()).is_file()}
    if baseline != snapshot["baseline_code_sha256"]:
        raise ValueError("The original HippoRAG Python sources changed")
    payload_sha = sha256(out / "repaired_openie.json")
    semantic = read_json(out / "semantic_summary.json")
    marker = read_json(out / "build_complete.json")
    if (semantic.get("complete") is not True
            or semantic.get("repaired_openie_sha256") != payload_sha
            or marker.get("repaired_openie_sha256") != payload_sha):
        raise ValueError("Source verification and built graph do not describe the current repaired OpenIE")
    return validation


def require_retrieval(out, validation):
    smoke = read_json(out / "retrieval_smoke_report.json")
    if (smoke.get("complete") is not True
            or smoke.get("graph_vectors_openie_unchanged") is not True
            or smoke.get("repaired_index") != validation["repaired_index"]
            or smoke.get("asset_sha256") != validation["asset_sha256"]
            or set(smoke.get("cases", {})) != {"hipporag2", "exp4_dependency_binding"}):
        raise ValueError("Both real retrieval smoke cases must pass on the exact validated index")
    for name, case in smoke["cases"].items():
        result_path = Path(case.get("result_path", "")).resolve()
        if (case.get("complete") is not True or case.get("n_samples") != 3
                or case.get("llm_request_stats", {}).get("failures", 0)
                or not result_path.is_relative_to(out / "retrieval_smoke")
                or not result_path.is_file()):
            raise ValueError(f"Invalid retrieval smoke artifact for {name}")
    return smoke


def run(args):
    out, hippo_root = Path(args.out_root).resolve(), Path(args.hippo_root).resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Completion artifacts must stay under PathCondRAG/outputs")
    out.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "stage": "starting", "out_root": str(out),
              "proposal_deleted": False, "started_epoch": time.time()}
    destination = out / "completion_report.json"
    # Only one finalizer may launch retrieval checks or remove the proposal.
    with (out / ".finish.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            report["stage"] = "waiting_for_repair"
            write_json(destination, report)
            waited = wait_for_repair(args.wait_pid, out) if args.wait_pid else None
            report["waited_repair"] = waited
            report["stage"] = "validating_index"
            write_json(destination, report)
            validation = require_validation(out, hippo_root, waited)
            environment = os.environ.copy()
            runtime = out / "runtime_deps"
            if runtime.is_dir():
                environment["PYTHONPATH"] = str(runtime) + (
                    os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
            environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
            report["stage"] = "retrieval_smoke"
            write_json(destination, report)
            subprocess.run([args.python, "-B", "-u", str(ROOT / "scripts/verify_openie_repaired_index.py"),
                            "--out-root", str(out), "--hippo-root", str(hippo_root),
                            "--datasets-dir", args.datasets_dir, "--llm-base-url", args.llm_base_url,
                            "--python", args.python], check=True, cwd=ROOT, env=environment)
            validation = require_validation(out, hippo_root, waited)
            smoke = require_retrieval(out, validation)
            head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
            report.update(repaired_index=validation["repaired_index"],
                          source_index_unchanged=True, baseline_code_unchanged=True,
                          asset_sha256=validation["asset_sha256"], git_head=head,
                          retrieval_cases=list(smoke["cases"]),
                          validation_report=str(out / "validation_report.json"),
                          retrieval_smoke_report=str(out / "retrieval_smoke_report.json"))
            if args.remove_proposal:
                existed = PROPOSAL.exists()
                PROPOSAL.unlink(missing_ok=True)
                report.update(proposal_deleted=existed, proposal_absent=not PROPOSAL.exists(),
                              proposal_path=str(PROPOSAL))
            report.update(complete=True, stage="complete", finished_epoch=time.time())
            write_json(destination, report)
            LOG.info("[done] repaired index and both retrievers verified; proposal_deleted=%s", report["proposal_deleted"])
            return report
        except Exception as error:
            report.update(complete=False, error=str(error), finished_epoch=time.time())
            write_json(destination, report)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(OUT_DEFAULT))
    parser.add_argument("--wait-pid", type=int, default=0, help="Wait for an existing, verified repair PID")
    parser.add_argument("--remove-proposal", action="store_true", help="Remove the requested proposal only after all checks")
    parser.add_argument("--python", default=RAG_PYTHON)
    parser.add_argument("--hippo-root", default="/root/baseline/HippoRAG")
    parser.add_argument("--datasets-dir", default="/root/datasets")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8035/v1")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)
    return 0
