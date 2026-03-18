import subprocess
import sys

import pgvector.psycopg
import psycopg

from ..base.module import BaseANN


class Meerkat(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._nlist = method_param["nlist"]
        self._boundary_epsilon = method_param.get("boundary_epsilon", 0)
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
        subprocess.run(
            "service postgresql start",
            shell=True, check=True,
            stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(
            user="ann", password="ann", dbname="ann", autocommit=True)
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
        if self._boundary_epsilon > 0:
            with_opts += ", boundary_epsilon = %g" % self._boundary_epsilon
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
        eps = (f", boundary_epsilon={self._boundary_epsilon}"
               if self._boundary_epsilon > 0 else "")
        return (f"Meerkat(metric={self._metric}, nlist={self._nlist},"
                f" nprobe={self._nprobe}, topk={self._topk}{eps})")
