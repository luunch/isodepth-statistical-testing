#!/bin/bash
# Submit one HT268B1 CalicoST slice (U1–U5) with 999 permutations.
#
# Usage:
#   ./run_calicost_ht268b1_u1_u5.sh U1
#   ./run_calicost_ht268b1_u1_u5.sh U3
#
# Or submit directly:
#   sbatch --export=ALL,CONFIG=configs/calicost/HT268B1_slice1_U1_loss_difference.json run_config.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

SLICE="${1:-}"
if [[ -z "${SLICE}" ]]; then
  echo "Usage: $0 {U1|U2|U3|U4|U5}" >&2
  exit 1
fi

case "${SLICE}" in
  U1) CONFIG="configs/calicost/HT268B1_slice1_U1_loss_difference.json" ;;
  U2) CONFIG="configs/calicost/HT268B1_slice3_U2_loss_difference.json" ;;
  U3) CONFIG="configs/calicost/HT268B1_slice2_U3_loss_difference.json" ;;
  U4) CONFIG="configs/calicost/HT268B1_slice4_U4_loss_difference.json" ;;
  U5) CONFIG="configs/calicost/HT268B1_slice5_U5_loss_difference.json" ;;
  *)
    echo "Unknown slice: ${SLICE} (expected U1–U5)" >&2
    exit 1
    ;;
esac

JID=$(sbatch --parsable --export=ALL,CONFIG="${CONFIG}" run_config.sh)
echo "submitted ${SLICE}: job ${JID}"
echo "config: ${CONFIG}"
echo "log:    isodepth_config-${JID}.log"
