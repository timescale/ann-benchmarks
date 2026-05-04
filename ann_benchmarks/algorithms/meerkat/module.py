import subprocess
import sys

import pgvector.psycopg
import psycopg

from ..base.module import BaseANN


class Meerkat(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._nlist = method_param["nlist"]
        self._fan_out = method_param.get("fan_out")
        self._centroid_compression = method_param.get("centroid_compression", False)
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
        with_opts = "nlist = %d" % self._nlist
        if self._fan_out is not None:
            with_opts += ", fan_out = %d" % self._fan_out
        if self._centroid_compression:
            with_opts += ", centroid_compression = true"
        if self._boundary_epsilon > 0:
            with_opts += ", boundary_epsilon = %g" % self._boundary_epsilon
        if self._soar_lambda > 0:
            with_opts += ", soar_lambda = %g" % self._soar_lambda
        cur.execute(
            "CREATE INDEX ON items USING mktann (embedding %s)"
            " WITH (%s)" % (self._ops, with_opts))
        print("done!")
        self._cur = cur

    def set_query_arguments(self, nprobe_topk):
        self._nprobe, self._topk = nprobe_topk
        self._cur.execute("SET mkt.nprobe = %d" % self._nprobe)

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
        parts = [f"metric={self._metric}", f"nlist={self._nlist}"]
        if self._fan_out is not None:
            parts.append(f"fan_out={self._fan_out}")
        if self._centroid_compression:
            parts.append("compress=true")
        parts.append(f"nprobe={self._nprobe}")
        parts.append(f"topk={self._topk}")
        if self._boundary_epsilon > 0:
            parts.append(f"boundary_epsilon={self._boundary_epsilon}")
        if self._soar_lambda > 0:
            parts.append(f"soar_lambda={self._soar_lambda}")
        return f"Meerkat({', '.join(parts)})"
