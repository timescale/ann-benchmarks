#!/usr/bin/env python3
import argparse
import csv

from ann_benchmarks.datasets import DATASETS, get_dataset
from ann_benchmarks.plotting.utils import compute_metrics_all_runs
from ann_benchmarks.results import load_all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute metrics")
    parser.add_argument("--batch", action="store_true",
                        help="Process batch mode results")
    args = parser.parse_args()

    datasets = DATASETS.keys()
    dfs = []
    for dataset_name in datasets:
        print("Looking at dataset", dataset_name)
        if len(list(load_all_results(dataset_name, batch_mode=args.batch))) > 0:
            results = load_all_results(dataset_name, batch_mode=args.batch)
            dataset, _ = get_dataset(dataset_name)
            results = compute_metrics_all_runs(
                dataset, results, args.recompute)
            for res in results:
                res["dataset"] = dataset_name
                dfs.append(res)
    if len(dfs) > 0:
        for res in dfs:
            print("%s recall=%.3f qps=%.0f shared_buffers=%.0f read=%.0f shared_buffers_per_query=%.0f p50=%.3f p99=%.3f" % (
                res["parameters"], res["k-nn"], res["qps"], res["shared_buffers"], res["shared_buffers_read"],  res["shared_buffers_per_query"], res["p50"], res["p99"]))
