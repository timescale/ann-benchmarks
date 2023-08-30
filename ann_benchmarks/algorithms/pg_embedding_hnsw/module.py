import subprocess
import sys

import psycopg

from ..base.module import BaseANN


class PGEmbeddingHNSW(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param["M"]
        self._ef_construction = method_param["efConstruction"]
        self._cur = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s::real[] LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s::real[] LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        #subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(user="ann", password="ann", dbname="ann", host="localhost")
        cur = conn.cursor()
        cur.execute("select count(*) from pg_class where relname = 'items'")
        table_count = cur.fetchone()[0]

        if table_count == 0:
            cur.execute("CREATE TABLE items (id int, embedding real[])")
            cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
            print("copying data...")
            with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
                for i, embedding in enumerate(X):
                    copy.write_row((i, embedding.tolist()))
        
        cur.execute("drop index if exists items_embedding_idx")
        cur.execute("select count(*) from pg_indexes where indexname = 'items_embedding_idx'")
        index_count = cur.fetchone()[0]

        if index_count == 0:
            print("Creating Index (dims=%d, m=%d, efconstruction=%d)" % (X.shape[1], self._m, self._ef_construction))
            if self._metric == "angular":
                cur.execute(
                    "CREATE INDEX items_embedding_idx ON items USING hnsw (embedding ann_cos_ops) WITH (dims=%d, m = %d, efconstruction = %d)" % (X.shape[1], self._m, self._ef_construction)
                )
            elif self._metric == "euclidean":
                cur.execute(
                    "CREATE INDEX items_embedding_idx ON items USING hnsw (embedding ann_l2_ops) WITH (dims=%d, m = %d, efconstruction = %d)" % (X.shape[1], self._m, self._ef_construction)
                    )
            else:
                raise RuntimeError(f"unknown metric {self._metric}")
            print("Index construction done!")
            #print("prewarming index...")
            #cur.execute("SELECT pg_prewarm('items_embedding_idx', 'buffer')")
            #print("Index prewarming done!")
        self._cur = cur

    def set_query_arguments(self, ef):
        self._ef = ef
        self._cur.execute("ALTER INDEX items_embedding_idx  SET (efsearch=%d)" % ef)
        print("ALTER INDEX items_embedding_idx SET (efsearch=%d)" % ef)
        # disable parallel query execution
        self._cur.execute("SET max_parallel_workers_per_gather = 0")

    def query(self, v, n):
        self._cur.execute(self._query, (v.tolist(), n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"PGEmbedding(m={self._m}, probes={self._ef_construction}, query_ef_search={self._ef})"
