#!/usr/bin/env python3
"""MuSiQue evaluation wrapper for PCRAG."""

from eval_dataset import build_parser, run_eval


def main():
    parser = build_parser()
    for action in parser._actions:
        if action.dest == "dataset":
            action.required = False
            action.default = "musique"
            break
    args = parser.parse_args()
    args.dataset = "musique"
    run_eval(dataset="musique", args=args)


if __name__ == "__main__":
    main()
