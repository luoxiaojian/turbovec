/**
 * RaBitQ flat-scan (bruteforce) benchmark on cohere dataset.
 *
 * Uses IVF with nlist=1, nprobe=1 to achieve exact flat scan through
 * all quantized codes — equivalent to a bruteforce index.
 *
 * Usage:
 *   ./rabitq_flat_bench <base.fvecs> <query.fvecs> <gt.ivecs>
 *                       <centroids_1.fvecs> <clusterids_1.ivecs>
 *                       <total_bits> [metric] [use_hacc]
 *
 *   total_bits: 1-9  (commonly 1,2,3,4,5,7)
 *   metric:     "ip" or "l2" (default: ip)
 *   use_hacc:   "true" or "false" (default: true)
 *
 * Build (from RaBitQ-Library root):
 *   g++ -std=c++17 -O3 -march=native -fopenmp -I include \
 *       rabitq_flat_bench.cpp -o rabitq_flat_bench
 */

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"

using PID = rabitqlib::PID;
using IVF = rabitqlib::ivf::IVF;
using DataMat = rabitqlib::RowMajorArray<float>;
using GtMat = rabitqlib::RowMajorArray<uint32_t>;

static constexpr size_t kTopK = 10;
static constexpr size_t kWarmupQueries = 10;
static constexpr size_t kTestRounds = 5;

static float compute_recall(
    const std::vector<PID>& results,
    const GtMat& ground_truth,
    size_t query_idx,
    size_t k
) {
    size_t hits = 0;
    size_t gt_k = std::min(k, static_cast<size_t>(ground_truth.cols()));
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < gt_k; ++j) {
            if (results[i] == ground_truth(query_idx, j)) {
                ++hits;
                break;
            }
        }
    }
    return static_cast<float>(hits) / static_cast<float>(k);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr
            << "Usage: " << argv[0]
            << " <base.fvecs> <query.fvecs> <gt.ivecs>"
            << " <centroids_1.fvecs> <clusterids_1.ivecs>"
            << " <total_bits> [metric=ip] [use_hacc=true]\n";
        return 1;
    }

    const char* base_file = argv[1];
    const char* query_file = argv[2];
    const char* gt_file = argv[3];
    const char* centroids_file = argv[4];
    const char* cids_file = argv[5];
    size_t total_bits = std::atoi(argv[6]);

    rabitqlib::MetricType metric_type = rabitqlib::METRIC_IP;
    if (argc > 7) {
        std::string metric_str(argv[7]);
        if (metric_str == "l2" || metric_str == "L2") {
            metric_type = rabitqlib::METRIC_L2;
        }
    }

    bool use_hacc = true;
    if (argc > 8) {
        std::string hacc_str(argv[8]);
        if (hacc_str == "false") {
            use_hacc = false;
        }
    }

    // Force single-threaded
    omp_set_num_threads(1);

    std::cout << "=== RaBitQ Flat Benchmark ===" << std::endl;
    std::cout << "  total_bits: " << total_bits << std::endl;
    std::cout << "  metric: " << (metric_type == rabitqlib::METRIC_IP ? "IP" : "L2")
              << std::endl;
    std::cout << "  use_hacc: " << (use_hacc ? "true" : "false") << std::endl;
    std::cout << "  top_k: " << kTopK << std::endl;

    // --- Load data ---
    DataMat base_data, centroids;
    GtMat ground_truth, cluster_ids;

    rabitqlib::load_vecs<float, DataMat>(base_file, base_data);
    rabitqlib::load_vecs<float, DataMat>(centroids_file, centroids);
    rabitqlib::load_vecs<PID, GtMat>(cids_file, cluster_ids);

    size_t num_base = base_data.rows();
    size_t dim = base_data.cols();
    size_t num_clusters = centroids.rows();  // should be 1

    std::cout << "  base: " << num_base << " x " << dim << std::endl;
    std::cout << "  clusters: " << num_clusters << std::endl;

    // --- Build index ---
    std::cout << "\n[1] Building index ..." << std::endl;
    auto build_start = std::chrono::high_resolution_clock::now();

    IVF ivf(num_base, dim, num_clusters, total_bits, metric_type);
    ivf.construct(base_data.data(), centroids.data(), cluster_ids.data(), false);

    auto build_end = std::chrono::high_resolution_clock::now();
    double build_sec =
        std::chrono::duration<double>(build_end - build_start).count();
    std::cout << "  build time: " << std::fixed << std::setprecision(2)
              << build_sec << "s" << std::endl;

    // --- Load queries and ground truth ---
    DataMat query_data;
    rabitqlib::load_vecs<float, DataMat>(query_file, query_data);
    rabitqlib::load_vecs<uint32_t, GtMat>(gt_file, ground_truth);

    size_t num_queries = query_data.rows();
    std::cout << "  queries: " << num_queries << std::endl;

    // --- Warmup ---
    std::cout << "\n[2] Warmup ..." << std::endl;
    {
        std::vector<PID> results(kTopK);
        size_t warmup_count = std::min(kWarmupQueries, num_queries);
        for (size_t i = 0; i < warmup_count; ++i) {
            ivf.search(&query_data(i, 0), kTopK, num_clusters, results.data(), use_hacc);
        }
    }

    // --- Benchmark ---
    std::cout << "\n[3] Benchmark (" << kTestRounds << " rounds) ..." << std::endl;

    std::vector<double> round_times(kTestRounds);
    float total_recall = 0;

    for (size_t round = 0; round < kTestRounds; ++round) {
        std::vector<PID> results(kTopK);
        float round_recall = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_queries; ++i) {
            ivf.search(
                &query_data(i, 0), kTopK, num_clusters, results.data(), use_hacc
            );
            // Compute recall only on last round to avoid timing overhead
            if (round == kTestRounds - 1) {
                round_recall += compute_recall(results, ground_truth, i, kTopK);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        round_times[round] = std::chrono::duration<double>(t1 - t0).count();

        if (round == kTestRounds - 1) {
            total_recall = round_recall / static_cast<float>(num_queries);
        }
    }

    // Use median time
    std::sort(round_times.begin(), round_times.end());
    double median_time = round_times[kTestRounds / 2];
    double qps = static_cast<double>(num_queries) / median_time;
    double ms_per_query = median_time / static_cast<double>(num_queries) * 1000.0;

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "  total_bits:    " << total_bits << std::endl;
    std::cout << "  recall@" << kTopK << ":    " << std::fixed
              << std::setprecision(4) << total_recall << std::endl;
    std::cout << "  QPS:           " << std::fixed << std::setprecision(1) << qps
              << std::endl;
    std::cout << "  ms/query:      " << std::fixed << std::setprecision(3)
              << ms_per_query << std::endl;
    std::cout << "  build_sec:     " << std::fixed << std::setprecision(2)
              << build_sec << std::endl;

    // Machine-readable output line
    std::cout << "\nCSV: rabitq," << total_bits << ","
              << std::fixed << std::setprecision(4) << total_recall << ","
              << std::fixed << std::setprecision(1) << qps << ","
              << std::fixed << std::setprecision(3) << ms_per_query << ","
              << std::fixed << std::setprecision(2) << build_sec << std::endl;

    return 0;
}
