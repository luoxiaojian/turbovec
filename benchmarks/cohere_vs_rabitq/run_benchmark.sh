#!/usr/bin/env bash
#
# Unified benchmark runner: TurboVec vs RaBitQ flat-scan on cohere datasets.
#
# Usage:
#   ./run_benchmark.sh [--size 1m|10m] [--data-dir ~/data/cohere]
#                      [--rabitq-dir ~/workspace/RaBitQ-Library]
#                      [--skip-download] [--skip-build]
#
# Prerequisites:
#   - Rust toolchain (cargo)
#   - C++17 compiler (g++ or clang++)
#   - Python 3 with numpy, h5py, faiss-cpu
#   - OpenBLAS (Linux) or Accelerate (macOS)
#
# What it does:
#   1. Download & prepare cohere dataset (HDF5 → fvecs/ivecs)
#   2. Build turbovec Rust benchmark
#   3. Build RaBitQ C++ benchmark
#   4. Run both at multiple bit widths, single-threaded
#   5. Output a comparison table with recall-aligned QPS
#
set -euo pipefail

# --- Defaults ---
SIZE="1m"
DATA_DIR="${HOME}/data/cohere"
RABITQ_DIR="${HOME}/workspace/RaBitQ-Library"
TURBOVEC_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SKIP_DOWNLOAD=false
SKIP_BUILD=false

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --size)       SIZE="$2"; shift 2;;
        --data-dir)   DATA_DIR="$2"; shift 2;;
        --rabitq-dir) RABITQ_DIR="$2"; shift 2;;
        --skip-download) SKIP_DOWNLOAD=true; shift;;
        --skip-build)    SKIP_BUILD=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

BENCH_DIR="${TURBOVEC_DIR}/benchmarks/cohere_vs_rabitq"
RESULTS_FILE="${BENCH_DIR}/results_cohere_${SIZE}.csv"

# Data file paths
BASE_FVECS="${DATA_DIR}/cohere_${SIZE}_base.fvecs"
QUERY_FVECS="${DATA_DIR}/cohere_${SIZE}_query.fvecs"
GT_IVECS="${DATA_DIR}/cohere_${SIZE}_groundtruth.ivecs"
CENTROIDS_FVECS="${DATA_DIR}/cohere_${SIZE}_centroids_1.fvecs"
CIDS_IVECS="${DATA_DIR}/cohere_${SIZE}_clusterids_1.ivecs"

echo "============================================="
echo "  TurboVec vs RaBitQ Flat-Scan Benchmark"
echo "  Dataset: cohere-${SIZE}"
echo "============================================="
echo ""

# =============================================
# Step 1: Download & prepare data
# =============================================
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "[Step 1] Downloading & preparing cohere-${SIZE} ..."
    python3 "${BENCH_DIR}/download_cohere.py" --size "${SIZE}" --data-dir "${DATA_DIR}"
    echo ""
else
    echo "[Step 1] Skipping download (--skip-download)"
    echo ""
fi

# Verify data files exist
for f in "$BASE_FVECS" "$QUERY_FVECS" "$GT_IVECS" "$CENTROIDS_FVECS" "$CIDS_IVECS"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing data file: $f"
        echo "  Run without --skip-download first."
        exit 1
    fi
done

# =============================================
# Step 2: Build benchmarks
# =============================================
if [ "$SKIP_BUILD" = false ]; then
    echo "[Step 2a] Building turbovec benchmark ..."
    cd "${TURBOVEC_DIR}"
    RAYON_NUM_THREADS=1 cargo build --release --example cohere_flat_bench 2>&1 | tail -3
    TURBOVEC_BIN="${TURBOVEC_DIR}/target/release/examples/cohere_flat_bench"
    echo "  Binary: ${TURBOVEC_BIN}"
    echo ""

    echo "[Step 2b] Building RaBitQ benchmark ..."
    RABITQ_BIN="${BENCH_DIR}/rabitq_flat_bench"
    CXX="${CXX:-g++}"
    ${CXX} -std=c++17 -O3 -march=native -fopenmp \
        -I "${RABITQ_DIR}/include" \
        "${BENCH_DIR}/rabitq_flat_bench.cpp" \
        -o "${RABITQ_BIN}"
    echo "  Binary: ${RABITQ_BIN}"
    echo ""
else
    echo "[Step 2] Skipping build (--skip-build)"
    TURBOVEC_BIN="${TURBOVEC_DIR}/target/release/examples/cohere_flat_bench"
    RABITQ_BIN="${BENCH_DIR}/rabitq_flat_bench"
    echo ""
fi

# =============================================
# Step 3: Run benchmarks
# =============================================
echo "[Step 3] Running benchmarks (single-threaded) ..."
echo ""

# CSV header
echo "engine,bits,recall@10,qps,ms_per_query,build_sec" > "${RESULTS_FILE}"

# --- TurboVec: bit_width 2, 3, 4 ---
TURBOVEC_BITS=(2 3 4)
for bw in "${TURBOVEC_BITS[@]}"; do
    echo "  >>> TurboVec ${bw}-bit ..."
    RAYON_NUM_THREADS=1 "${TURBOVEC_BIN}" \
        "${BASE_FVECS}" "${QUERY_FVECS}" "${GT_IVECS}" "${bw}" \
        2>/dev/null | grep "^CSV:" | sed 's/^CSV: //' >> "${RESULTS_FILE}"
    echo ""
done

# --- RaBitQ: total_bits 1, 2, 3, 4, 5, 7 ---
RABITQ_BITS=(1 2 3 4 5 7)
for bits in "${RABITQ_BITS[@]}"; do
    echo "  >>> RaBitQ ${bits}-bit ..."
    OMP_NUM_THREADS=1 "${RABITQ_BIN}" \
        "${BASE_FVECS}" "${QUERY_FVECS}" "${GT_IVECS}" \
        "${CENTROIDS_FVECS}" "${CIDS_IVECS}" \
        "${bits}" ip true \
        2>/dev/null | grep "^CSV:" | sed 's/^CSV: //' >> "${RESULTS_FILE}"
    echo ""
done

# =============================================
# Step 4: Print comparison table
# =============================================
echo ""
echo "============================================="
echo "  Results: cohere-${SIZE}"
echo "============================================="
echo ""

# Print CSV as formatted table
printf "%-10s %5s %10s %10s %10s %10s\n" "Engine" "Bits" "Recall@10" "QPS" "ms/q" "Build(s)"
printf "%-10s %5s %10s %10s %10s %10s\n" "------" "----" "---------" "---" "----" "--------"

# Skip header, sort by engine then bits
tail -n +2 "${RESULTS_FILE}" | sort -t',' -k1,1 -k2,2n | while IFS=',' read -r engine bits recall qps msq build; do
    printf "%-10s %5s %10s %10s %10s %10s\n" "$engine" "$bits" "$recall" "$qps" "$msq" "$build"
done

echo ""
echo "Raw CSV: ${RESULTS_FILE}"
echo ""

# =============================================
# Step 5: Recall-aligned QPS comparison
# =============================================
echo "============================================="
echo "  Recall-Aligned QPS Comparison"
echo "============================================="
echo ""

python3 - "${RESULTS_FILE}" << 'PYEOF'
import csv
import sys

results_file = sys.argv[1]

turbovec_rows = []
rabitq_rows = []

with open(results_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        entry = {
            "bits": int(row["bits"]),
            "recall": float(row["recall@10"]),
            "qps": float(row["qps"]),
            "ms_per_query": float(row["ms_per_query"]),
        }
        if row["engine"] == "turbovec":
            turbovec_rows.append(entry)
        else:
            rabitq_rows.append(entry)

if not turbovec_rows or not rabitq_rows:
    print("  Not enough data for comparison.")
    sys.exit(0)

# For each turbovec config, find the RaBitQ config with the closest recall
# that is >= turbovec's recall (i.e., RaBitQ at equal or better recall)
print(f"  {'TQ bits':>7} {'TQ R@10':>8} {'TQ QPS':>10} | "
      f"{'RQ bits':>7} {'RQ R@10':>8} {'RQ QPS':>10} | {'Speedup':>8}")
print(f"  {'-'*7} {'-'*8} {'-'*10} | {'-'*7} {'-'*8} {'-'*10} | {'-'*8}")

for tq in sorted(turbovec_rows, key=lambda x: x["bits"]):
    # Find RaBitQ configs with recall >= tq recall (within 1% tolerance)
    target_recall = tq["recall"] - 0.01
    candidates = [rq for rq in rabitq_rows if rq["recall"] >= target_recall]
    if not candidates:
        # Fall back: pick the one with highest recall
        candidates = rabitq_rows

    # Among candidates, pick the one with closest recall
    best_rq = min(candidates, key=lambda rq: abs(rq["recall"] - tq["recall"]))

    speedup = tq["qps"] / best_rq["qps"] if best_rq["qps"] > 0 else float("inf")
    print(f"  {tq['bits']:>7} {tq['recall']:>8.4f} {tq['qps']:>10.1f} | "
          f"{best_rq['bits']:>7} {best_rq['recall']:>8.4f} {best_rq['qps']:>10.1f} | "
          f"{speedup:>7.2f}x")

print()
PYEOF

echo "Done!"
