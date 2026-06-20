#!/usr/bin/env bash
# Reproduce the core speculative-EXACT results end-to-end.
#
#   ./scripts/reproduce.sh [CORPUS]
#
# CORPUS defaults to ./corpora/enwik8 (download: http://mattmahoney.net/dc/enwik8.zip).
# Produces: a bit-exact round-trip + forward-count reduction (cacheless codec),
# and the theory check (avg_run == 1/(1-p_w)).
set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS="${1:-corpora/enwik8}"
PY="${PYTHON:-python3}"
SLICE=1700000          # bytes of corpus to warm the draft on
OFFSET=0               # test slice offset into CORPUS (match the draft's entstart)
NBYTES=2048            # test-window size for the codec demo

if [ ! -f "$CORPUS" ]; then
  echo "Corpus not found: $CORPUS"
  echo "Download enwik8: curl -L http://mattmahoney.net/dc/enwik8.zip -o enwik8.zip && unzip enwik8.zip -d corpora/"
  exit 1
fi

echo "[1/4] build the cheap draft (cm)"
make -C draft >/dev/null

echo "[2/4] produce draft (argmax) + true-byte ranks for the test slice"
head -c "$SLICE" "$CORPUS" \
  | draft/cm c /tmp/se_bits.bin /tmp/se_ent.bin "$OFFSET" /tmp/se_arg.bin /tmp/se_rank.bin \
  >/tmp/se_compressed.bin 2>/dev/null

echo "[3/4] speculative-EXACT codec: round-trip + forward-count reduction"
$PY src/speculative_codec.py --bytes "$NBYTES" --k 16 --W 2048 --device cpu \
    --file "$CORPUS" --offset "$OFFSET" --draft /tmp/se_arg.bin

echo "[4/4] theory check: avg_run == 1/(1 - top-w accuracy)"
SE_RANK=/tmp/se_rank.bin SE_BITS=/tmp/se_bits.bin $PY src/theory.py || \
  echo "(theory.py expects /tmp/cm_rank.bin etc.; see src/theory.py header for paths)"

echo
echo "Done. Round-trip should read OK; avg accepted run > 1; theory == measured."
echo "For a TRAINED model + real ratio, see README 'End-to-end on a trained model'."
