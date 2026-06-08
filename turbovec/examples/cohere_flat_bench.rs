//! Turbovec flat-scan (bruteforce) benchmark on cohere dataset.
//!
//! Reads fvecs/ivecs files, builds a TurboQuantIndex, and measures
//! single-threaded QPS at various bit widths.
//!
//! ```text
//! cargo run --release --example cohere_flat_bench -- \
//!     ~/data/cohere/cohere_1m_base.fvecs \
//!     ~/data/cohere/cohere_1m_query.fvecs \
//!     ~/data/cohere/cohere_1m_groundtruth.ivecs \
//!     <bit_width>
//! ```

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Instant;

use turbovec::TurboQuantIndex;

const TOP_K: usize = 10;
const WARMUP_QUERIES: usize = 10;
const TEST_ROUNDS: usize = 5;

/// Read an fvecs file: each row is [dim as u32] [dim x f32].
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let file_len = file.metadata().unwrap().len() as usize;
    let mut reader = BufReader::new(file);

    // Read dimension from first record
    let mut dim_buf = [0u8; 4];
    reader.read_exact(&mut dim_buf).unwrap();
    let dim = u32::from_le_bytes(dim_buf) as usize;

    let row_bytes = 4 + dim * 4; // 4 bytes dim + dim floats
    let num_rows = file_len / row_bytes;

    // Reset and read all
    let mut data = vec![0f32; num_rows * dim];
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);

    for i in 0..num_rows {
        // Skip dim field
        reader.read_exact(&mut dim_buf).unwrap();
        let row_slice = &mut data[i * dim..(i + 1) * dim];
        let byte_slice = unsafe {
            std::slice::from_raw_parts_mut(row_slice.as_mut_ptr() as *mut u8, dim * 4)
        };
        reader.read_exact(byte_slice).unwrap();
    }

    eprintln!("  Loaded {path}: {num_rows} x {dim}");
    (data, num_rows, dim)
}

/// Read an ivecs file: each row is [k as u32] [k x i32].
fn read_ivecs(path: &str) -> (Vec<i32>, usize, usize) {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let file_len = file.metadata().unwrap().len() as usize;
    let mut reader = BufReader::new(file);

    let mut dim_buf = [0u8; 4];
    reader.read_exact(&mut dim_buf).unwrap();
    let k = u32::from_le_bytes(dim_buf) as usize;

    let row_bytes = 4 + k * 4;
    let num_rows = file_len / row_bytes;

    let mut data = vec![0i32; num_rows * k];
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);

    for i in 0..num_rows {
        reader.read_exact(&mut dim_buf).unwrap();
        let row_slice = &mut data[i * k..(i + 1) * k];
        let byte_slice = unsafe {
            std::slice::from_raw_parts_mut(row_slice.as_mut_ptr() as *mut u8, k * 4)
        };
        reader.read_exact(byte_slice).unwrap();
    }

    eprintln!("  Loaded {path}: {num_rows} x {k}");
    (data, num_rows, k)
}

fn compute_recall(
    result_indices: &[i64],
    ground_truth: &[i32],
    gt_k: usize,
    query_idx: usize,
    k: usize,
) -> f64 {
    let gt_row = &ground_truth[query_idx * gt_k..query_idx * gt_k + k.min(gt_k)];
    let res_row = &result_indices[..k];
    let mut hits = 0usize;
    for &r in res_row {
        for &g in gt_row {
            if r == g as i64 {
                hits += 1;
                break;
            }
        }
    }
    hits as f64 / k as f64
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "Usage: {} <base.fvecs> <query.fvecs> <gt.ivecs> <bit_width>",
            args[0]
        );
        std::process::exit(1);
    }

    let base_path = &args[1];
    let query_path = &args[2];
    let gt_path = &args[3];
    let bit_width: usize = args[4].parse().expect("bit_width must be 2, 3, or 4");

    // Force single thread
    env::set_var("RAYON_NUM_THREADS", "1");

    eprintln!("=== TurboVec Flat Benchmark ===");
    eprintln!("  bit_width: {bit_width}");
    eprintln!("  top_k: {TOP_K}");

    // --- Load data ---
    let (base_data, num_base, dim) = read_fvecs(base_path);
    let (query_data, num_queries, _query_dim) = read_fvecs(query_path);
    let (gt_data, _gt_rows, gt_k) = read_ivecs(gt_path);

    eprintln!("  base: {num_base} x {dim}");

    // --- Build index ---
    eprintln!("\n[1] Building index ...");
    let build_start = Instant::now();

    let mut index = TurboQuantIndex::new(dim, bit_width).expect("Failed to create index");
    index.add(&base_data);
    index.prepare();

    let build_sec = build_start.elapsed().as_secs_f64();
    eprintln!("  build time: {build_sec:.2}s");

    // --- Warmup ---
    eprintln!("\n[2] Warmup ...");
    {
        let warmup_count = WARMUP_QUERIES.min(num_queries);
        let warmup_queries = &query_data[..warmup_count * dim];
        let _ = index.search(warmup_queries, TOP_K);
    }

    // --- Benchmark (one query at a time for fair single-thread comparison) ---
    eprintln!("\n[3] Benchmark ({TEST_ROUNDS} rounds) ...");

    let mut round_times = vec![0f64; TEST_ROUNDS];
    let mut total_recall = 0f64;

    for round in 0..TEST_ROUNDS {
        let t0 = Instant::now();
        let mut round_recall = 0f64;

        for qi in 0..num_queries {
            let query_slice = &query_data[qi * dim..(qi + 1) * dim];
            let results = index.search(query_slice, TOP_K);

            // Compute recall only on last round
            if round == TEST_ROUNDS - 1 {
                let result_indices = results.indices_for_query(0);
                round_recall +=
                    compute_recall(result_indices, &gt_data, gt_k, qi, TOP_K);
            }
        }

        round_times[round] = t0.elapsed().as_secs_f64();

        if round == TEST_ROUNDS - 1 {
            total_recall = round_recall / num_queries as f64;
        }
    }

    // Median time
    round_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_time = round_times[TEST_ROUNDS / 2];
    let qps = num_queries as f64 / median_time;
    let ms_per_query = median_time / num_queries as f64 * 1000.0;

    eprintln!("\n=== Results ===");
    eprintln!("  bit_width:     {bit_width}");
    eprintln!("  recall@{TOP_K}:    {total_recall:.4}");
    eprintln!("  QPS:           {qps:.1}");
    eprintln!("  ms/query:      {ms_per_query:.3}");
    eprintln!("  build_sec:     {build_sec:.2}");

    // Machine-readable CSV line on stdout
    println!(
        "CSV: turbovec,{bit_width},{total_recall:.4},{qps:.1},{ms_per_query:.3},{build_sec:.2}"
    );
}
