#!/usr/bin/env python3
"""Run normal evaluation with exact upstream evidence capture or replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-evidence-inputs", type=Path)
    mode.add_argument("--replay-evidence-inputs", type=Path)
    args, remaining = parser.parse_known_args(argv)
    # The original parser handles every experiment option; the wrapper only
    # requires an explicit output so the snapshot can bind that exact result.
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", required=True)
    output_parser.add_argument("--eval_mode", default="retrieve")
    output, _ = output_parser.parse_known_args(remaining)
    if output.eval_mode != "retrieve":
        raise ValueError("Frozen evidence experiments support retrieval evaluation only")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from pathcondrag.utils.evidence_frozen_inputs import frozen_evidence_runtime, experiment_metadata
    previous = sys.argv
    sys.argv = [str(root / "scripts/eval_dataset.py"), *remaining]
    try:
        with frozen_evidence_runtime(capture=args.capture_evidence_inputs,
                                     replay=args.replay_evidence_inputs) as collector:
            runpy.run_path(sys.argv[0], run_name="__main__")
        if collector is not None:
            collector.write(args.capture_evidence_inputs, output.output)
        else:
            manifest = json.loads((args.replay_evidence_inputs / "manifest.json").read_text())
            result_path = Path(output.output)
            result = json.loads(result_path.read_text())
            if result.get("selected_indices") != manifest["selected_indices"]:
                raise ValueError("Replay physical sample indices differ from capture")
            result["frozen_evidence_inputs"] = experiment_metadata(manifest, "replay")
            temporary = result_path.with_name(result_path.name + ".frozen.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(result_path)
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
