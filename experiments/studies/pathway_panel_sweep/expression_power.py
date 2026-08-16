from __future__ import annotations

import argparse
import json

from experiments.studies.pathway_panel_sweep.pathway_expression_power import (
    analyze_pathway_expression_power,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare expression depth / detection rate for existence-significant vs "
            "nonsignificant Hallmark pathways."
        ),
    )
    parser.add_argument("--spec", required=True, help="Path to the pathway sweep spec JSON")
    parser.add_argument(
        "--reference-pathway",
        default="HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        help="Reference isodepth pathway for spatial-gradient checks",
    )
    args = parser.parse_args()
    payload = analyze_pathway_expression_power(
        args.spec,
        reference_pathway_name=str(args.reference_pathway),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
