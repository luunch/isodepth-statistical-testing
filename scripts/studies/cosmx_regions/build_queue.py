"""Build the run queue for the CosMx cell-type region sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "cosmx_celltype_regions"
RESULTS = ROOT / "results" / "cosmx_celltype_regions"
QUEUE = RESULTS / "_run_queue.txt"


def _run_name(cfg: Path) -> str:
    return json.load(open(cfg)).get("output", {}).get("run_name", cfg.stem)


def _is_done(cfg: Path) -> bool:
    run_name = _run_name(cfg)
    return (RESULTS / run_name / f"{run_name}_result.json").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--rerun-all", action="store_true")
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("--queue-path", type=Path, default=QUEUE)
    args = ap.parse_args()

    all_cfgs = sorted(CONFIG_DIR.glob("*.json"))
    if not all_cfgs:
        raise SystemExit(f"no configs in {CONFIG_DIR} "
                         "(run scripts/segment_cosmx_celltype_regions.py first)")

    done = [c for c in all_cfgs if not args.rerun_all and _is_done(c)]
    pending = [c for c in all_cfgs if args.rerun_all or not _is_done(c)]
    selected = pending if args.limit <= 0 else pending[:args.limit]

    print(f"total regions: {len(all_cfgs)}")
    print(f"already done:  {len(done)}")
    print(f"pending:       {len(pending)}")
    print(f"limit:         {'all' if args.limit <= 0 else args.limit}")
    print(f"=> queuing:    {len(selected)}")

    if args.print_only:
        for c in selected:
            print(f"  {c.relative_to(ROOT)}")
        return

    args.queue_path.parent.mkdir(parents=True, exist_ok=True)
    args.queue_path.write_text("".join(f"{c.relative_to(ROOT)}\n" for c in selected))
    print(f"wrote {args.queue_path.relative_to(ROOT)} ({len(selected)} configs)")


if __name__ == "__main__":
    main()
