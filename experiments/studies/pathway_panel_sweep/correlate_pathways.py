from __future__ import annotations

import argparse
import json

from experiments.studies.pathway_panel_sweep.pathway_isodepth_correlation import (
    analyze_pathway_isodepth_correlations,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Spearman-correlate Hallmark pathway mean scores with a shared reference isodepth "
            "and with each pathway's own fitted isodepth."
        ),
    )
    parser.add_argument("--spec", required=True, help="Path to the pathway sweep spec JSON")
    parser.add_argument(
        "--reference-pathway",
        default="HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        help="Pathway whose fitted isodepth defines the shared reference axis",
    )
    parser.add_argument(
        "--expression-top-var-genes",
        type=int,
        default=3000,
        help="HVG count when building the broad expression matrix for pathway scores",
    )
    parser.add_argument(
        "--pathway-score-min-genes",
        type=int,
        default=5,
        help="Minimum matched genes required to score a pathway",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_pathway_isodepth_correlations(
        args.spec,
        reference_pathway_name=str(args.reference_pathway),
        expression_top_var_genes=int(args.expression_top_var_genes),
        pathway_score_min_genes=int(args.pathway_score_min_genes),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
