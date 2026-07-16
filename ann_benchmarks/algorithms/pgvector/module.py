import subprocess
import sys

import pgvector.psycopg
import psycopg

from ..base.module import BaseANN


class PGVector(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        # Half-precision expression index (embedding::halfvec(dim)):
        # ~2x smaller graph, which is what lets a 100M-vector build fit
        # in memory. The query is finalized in fit() where the
        # dimension is known.
        self._halfvec = method_param.get('halfvec', False)
        self._cur = None

        if metric == "angular":
            self._op = "<=>"
            self._ops_prefix = "halfvec_cosine_ops" if self._halfvec else "vector_cosine_ops"
        elif metric == "euclidean":
            self._op = "<->"
            self._ops_prefix = "halfvec_l2_ops" if self._halfvec else "vector_l2_ops"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(user="ann", password="ann", dbname="ann", autocommit=True)
        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print("creating index...")
        dim = X.shape[1]
        if self._halfvec:
            index_target = "(embedding::halfvec(%d))" % dim
            order_expr = "embedding::halfvec(%d) %s %%s::halfvec(%d)" % (dim, self._op, dim)
        else:
            index_target = "embedding"
            order_expr = "embedding %s %%s" % self._op
        self._query = "SELECT id FROM items ORDER BY " + order_expr + " LIMIT %s"
        cur.execute(
            "CREATE INDEX ON items USING hnsw (%s %s) WITH (m = %d, ef_construction = %d)"
            % (index_target, self._ops_prefix, self._m, self._ef_construction)
        )
        print("done!")
        self._cur = cur

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw.ef_search = %d" % ef_search)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        # Expression (halfvec) indexes get an auto-generated name, so
        # size whatever index exists on the table.
        self._cur.execute(
            "SELECT sum(pg_relation_size(indexrelid)) FROM pg_index "
            "WHERE indrelid = 'items'::regclass")
        return self._cur.fetchone()[0] / 1024

    def __str__(self):
        variant = ", halfvec" if self._halfvec else ""
        return f"PGVector(m={self._m}, ef_construction={self._ef_construction}{variant}, ef_search={self._ef_search})"
