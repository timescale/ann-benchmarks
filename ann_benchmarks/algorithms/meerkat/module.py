import subprocess
import sys

import pgvector.psycopg
import psycopg

from ..base.module import BaseANN

# Parallel worker budget for the index build, taken from the image's
# max_parallel_maintenance_workers: the standard image sets 0 (serial
# builds, the ann-benchmarks convention) and the large-dataset image
# opts in at docker build time. Meerkat derives its build
# worker count from the planner's heap-size heuristic, which derates
# large builds; pinning the table's parallel_workers reloption lets the
# build use the whole budget.


class Meerkat(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        # Every index parameter is optional. An index built WITHOUT
        # options resolves to meerkat's tuned defaults (auto nlist =
        # rows/256, fastscan posting + centroid layouts, SOAR and
        # boundary replication), which is the recommended configuration
        # at every scale -- the empty arg_groups entry in config.yml is
        # the primary benchmark arm.
        self._nlist = method_param.get("nlist")
        self._fan_out = method_param.get("fan_out")
        self._centroid_compression = method_param.get("centroid_compression", False)
        self._centroid_fastscan = method_param.get("centroid_fastscan", False)
        self._rerank_vectors = method_param.get("rerank_vectors", False)
        self._fastscan = method_param.get("fastscan", False)
        self._fastscan_bits = method_param.get("fastscan_bits")
        self._boundary_epsilon = method_param.get("boundary_epsilon", 0)
        self._soar_lambda = method_param.get("soar_lambda", 0)
        self._cur = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding OPERATOR(mkt.<=>) %s LIMIT %s"
            self._ops = "mkt.vector_cosine_ops"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding OPERATOR(mkt.<->) %s LIMIT %s"
            self._ops = "mkt.vector_l2_ops"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        import os
        dsn = os.environ.get("MEERKAT_DSN", "user=ann password=ann dbname=ann")
        if "MEERKAT_DSN" not in os.environ:
            subprocess.run(
                "service postgresql start",
                shell=True, check=True,
                stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(dsn, autocommit=True)
        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute(
            "CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute(
            "ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")

        print("copying data...")
        sys.stdout.flush()
        with cur.copy(
            "COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))

        print("creating index...")
        sys.stdout.flush()
        cur.execute("SHOW max_parallel_maintenance_workers")
        build_workers = int(cur.fetchone()[0])
        if build_workers > 0:
            cur.execute(
                "ALTER TABLE items SET (parallel_workers = %d)"
                % build_workers)
        with_opts = []
        if self._nlist is not None:
            with_opts.append("nlist = %d" % self._nlist)
        if self._fan_out is not None:
            with_opts.append("fan_out = %d" % self._fan_out)
        if self._centroid_compression:
            with_opts.append("centroid_compression = true")
        if self._centroid_fastscan:
            with_opts.append("centroid_fastscan = true")
        if self._rerank_vectors:
            with_opts.append("rerank_vectors = true")
        if self._fastscan:
            with_opts.append("fastscan = true")
        if self._boundary_epsilon > 0:
            with_opts.append("boundary_epsilon = %g" % self._boundary_epsilon)
        if self._soar_lambda > 0:
            with_opts.append("soar_lambda = %g" % self._soar_lambda)
        with_clause = " WITH (%s)" % ", ".join(with_opts) if with_opts else ""
        cur.execute(
            "CREATE INDEX ON items USING mktann (embedding %s)%s"
            % (self._ops, with_clause))
        if build_workers > 0:
            cur.execute("ALTER TABLE items RESET (parallel_workers)")
        print("done!")
        self._cur = cur

    def set_query_arguments(self, nprobe_topk):
        # nprobe = 0 keeps meerkat's automatic derivation
        # (~0.5*sqrt(nlist), targeting roughly 0.95 recall@10).
        self._nprobe, self._topk = nprobe_topk
        self._cur.execute("SET mkt.nprobe = %d" % self._nprobe)
        if self._fastscan_bits is not None:
            self._cur.execute("SET mkt.fastscan_bits = %d" % self._fastscan_bits)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute(
            "SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024

    def __str__(self):
        params = [f"nlist={self._nlist if self._nlist is not None else 'auto'}"]
        if self._fastscan:
            params.append(f"fs={self._fastscan_bits or 'default'}")
        if self._soar_lambda > 0:
            params.append(f"soar={self._soar_lambda}")
        if self._boundary_epsilon > 0:
            params.append(f"bε={self._boundary_epsilon}")
        if self._rerank_vectors:
            params.append("rerank")
        return (f"Meerkat (PG) [{', '.join(params)}]"
                f" nprobe={self._nprobe if self._nprobe else 'auto'}")
