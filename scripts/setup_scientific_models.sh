#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  Scientific Models — HuggingFace Cache Symlinks
# ════════════════════════════════════════════════════════════════════════
#
#  Run this ONCE on the login node before deploying with Scientific support.
#
#  Purpose:
#    The scientific pipeline needs 3 models (ColPali, SciNCL, Qwen2-VL-2B)
#    which are cached in Vineet's HF directory. The production server uses
#    Gokul's HF_HOME. This script creates symlinks so both caches coexist.
#
#  Usage:
#    bash scripts/setup_scientific_models.sh
#
#  This is idempotent — safe to run multiple times.
# ════════════════════════════════════════════════════════════════════════

set -e

SRC_HUB="/scratch/data/divyasaxena_rs/Vineet_internship/.cache/huggingface/hub"
DST_HUB="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.cache/huggingface/hub"

MODELS=(
    "models--vidore--colpali-v1.2"
    "models--malteos--scincl"
    "models--Qwen--Qwen2-VL-2B-Instruct"
)

echo "═══════════════════════════════════════════════════════════"
echo "  Scientific Models — HuggingFace Cache Symlinks"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Source: ${SRC_HUB}"
echo "  Target: ${DST_HUB}"
echo ""

# Ensure target hub directory exists
mkdir -p "${DST_HUB}"

CREATED=0
SKIPPED=0
ERRORS=0

for model in "${MODELS[@]}"; do
    SRC="${SRC_HUB}/${model}"
    DST="${DST_HUB}/${model}"

    if [ -L "${DST}" ]; then
        echo "  ✓ ${model} — symlink exists ($(readlink -f "${DST}"))"
        SKIPPED=$((SKIPPED + 1))
    elif [ -d "${DST}" ]; then
        echo "  ✓ ${model} — real directory exists (skipping)"
        SKIPPED=$((SKIPPED + 1))
    elif [ -d "${SRC}" ]; then
        ln -s "${SRC}" "${DST}"
        echo "  ✓ ${model} — symlink CREATED"
        CREATED=$((CREATED + 1))
    else
        echo "  ✗ ${model} — source NOT FOUND at ${SRC}"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Results: ${CREATED} created, ${SKIPPED} skipped, ${ERRORS} errors"
echo "═══════════════════════════════════════════════════════════"

if [ ${ERRORS} -gt 0 ]; then
    echo ""
    echo "  ⚠ Some models are missing from the source cache."
    echo "    They will need to download on first use (requires internet)."
    exit 1
fi

echo ""
echo "  ✅ All scientific models are available in ${DST_HUB}"
echo "  You can now run: sbatch scripts/slurm_scientific_validation.sh"
echo ""
