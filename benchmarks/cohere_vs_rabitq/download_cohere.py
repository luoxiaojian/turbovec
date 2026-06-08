#!/usr/bin/env python3
"""Download cohere-1m / cohere-10m datasets and convert to fvecs/ivecs for RaBitQ.

Usage:
    python3 download_cohere.py --size 1m   [--data-dir ~/data/cohere]
    python3 download_cohere.py --size 10m  [--data-dir ~/data/cohere]

Outputs (in data-dir):
    cohere_{size}_base.fvecs      — database vectors (L2-normalized)
    cohere_{size}_query.fvecs     — query vectors (L2-normalized)
    cohere_{size}_groundtruth.ivecs — ground truth (top-100 by IP on normalized vecs)
    cohere_{size}_centroids_1.fvecs — single zero centroid for flat-scan IVF
    cohere_{size}_clusterids_1.ivecs — all vectors assigned to cluster 0
"""
import argparse
import os
import struct
import sys

import numpy as np


def download_hdf5(size, data_dir):
    """Download cohere HDF5 from HuggingFace."""
    import urllib.request

    urls = {
        "1m": "https://huggingface.co/datasets/erikbern/ann-benchmarks/resolve/main/cohere-768-1m-euclidean.hdf5",
        "10m": "https://huggingface.co/datasets/makneeee/cohere_large_10m/resolve/main/vectors.hdf5",
    }
    hdf5_path = os.path.join(data_dir, f"cohere_{size}.hdf5")
    if os.path.exists(hdf5_path):
        print(f"  Already exists: {hdf5_path}")
        return hdf5_path

    url = urls[size]
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, hdf5_path)
    print(f"  Saved: {hdf5_path} ({os.path.getsize(hdf5_path) / 1024 / 1024:.0f} MB)")
    return hdf5_path


def load_hdf5(hdf5_path, size):
    """Load database and query vectors from HDF5."""
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        print(f"  HDF5 keys: {list(f.keys())}")
        database = np.array(f["train"], dtype=np.float32)
        queries = np.array(f["test"], dtype=np.float32)
    print(f"  database: {database.shape}, queries: {queries.shape}")
    return database, queries


def normalize(vectors):
    """L2-normalize, replacing zero-norm vectors with uniform direction."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    zero_mask = norms.squeeze() < 1e-10
    if zero_mask.any():
        print(f"  warning: {zero_mask.sum()} zero-norm vectors, replacing with uniform")
        vectors[zero_mask] = 1.0
        norms[zero_mask] = np.sqrt(vectors.shape[1])
    vectors /= norms
    return vectors


def compute_groundtruth(database, queries, k=100):
    """Compute exact top-k by inner product using faiss."""
    import faiss

    print(f"  Computing ground truth (FlatIP, k={k}) ...")
    index = faiss.IndexFlatIP(database.shape[1])
    index.add(database)
    _, indices = index.search(queries, k)
    return indices.astype(np.int32)


def write_fvecs(path, data):
    """Write float32 matrix to fvecs format."""
    n, d = data.shape
    with open(path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<i", d))
            f.write(data[i].tobytes())
    print(f"  Wrote {path} ({n} x {d})")


def write_ivecs(path, data):
    """Write int32 matrix to ivecs format."""
    n, d = data.shape
    with open(path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<i", d))
            f.write(data[i].astype(np.int32).tobytes())
    print(f"  Wrote {path} ({n} x {d})")


def write_clustering_files(data_dir, size, dim, n_vectors):
    """Write single-cluster centroid and cluster-id files for flat-scan IVF."""
    centroid = np.zeros((1, dim), dtype=np.float32)
    centroid_path = os.path.join(data_dir, f"cohere_{size}_centroids_1.fvecs")
    write_fvecs(centroid_path, centroid)

    cids = np.zeros((n_vectors, 1), dtype=np.int32)
    cids_path = os.path.join(data_dir, f"cohere_{size}_clusterids_1.ivecs")
    write_ivecs(cids_path, cids)


def main():
    parser = argparse.ArgumentParser(description="Download & prepare cohere dataset")
    parser.add_argument("--size", choices=["1m", "10m"], required=True)
    parser.add_argument("--data-dir", default=os.path.expanduser("~/data/cohere"))
    parser.add_argument("--k", type=int, default=100, help="Ground truth top-k")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    print(f"=== Preparing cohere-{args.size} ===")

    hdf5_path = download_hdf5(args.size, args.data_dir)
    database, queries = load_hdf5(hdf5_path, args.size)
    dim = database.shape[1]

    print("  Normalizing ...")
    database = normalize(database)
    queries = normalize(queries)

    gt = compute_groundtruth(database, queries, args.k)

    base_path = os.path.join(args.data_dir, f"cohere_{args.size}_base.fvecs")
    query_path = os.path.join(args.data_dir, f"cohere_{args.size}_query.fvecs")
    gt_path = os.path.join(args.data_dir, f"cohere_{args.size}_groundtruth.ivecs")

    print("  Writing fvecs/ivecs ...")
    write_fvecs(base_path, database)
    write_fvecs(query_path, queries)
    write_ivecs(gt_path, gt)
    write_clustering_files(args.data_dir, args.size, dim, database.shape[0])

    print(f"\n  Done! Files in {args.data_dir}")


if __name__ == "__main__":
    main()
