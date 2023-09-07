import subprocess
import sys

import pgvector.psycopg
import psycopg

from ..base.module import BaseANN

class PGVectorHNSW(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param["M"]
        self._ef_construction = method_param["efConstruction"]
        self._cur = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        #subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(user="ann", password="ann", dbname="ann", host="localhost")
        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("select count(*) from pg_class where relname = 'items'")
        table_count = cur.fetchone()[0]
        if table_count == 0:
            cur.execute("CREATE TABLE IF NOT EXISTS items (id int, embedding vector(%d))" % X.shape[1])
            cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
            print("copying data...")
            with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
                for i, embedding in enumerate(X):
                    copy.write_row((i, embedding))
        
        cur.execute("drop index if exists pgv_idx")
        cur.execute("select count(*) from pg_indexes where indexname = 'pgv_idx'")
        index_count = cur.fetchone()[0]

        if index_count == 0:
            print("Creating Index (m = %s, ef_construction = %d)" % (self._m, self._ef_construction))
            cur.execute("SET maintenance_work_mem = '8GB'")
            cur.execute("SET min_parallel_table_scan_size TO 1")
            if self._metric == "angular":
                cur.execute(
                    "CREATE INDEX pgv_idx ON items USING hnsw (embedding vector_cosine_ops) WITH (m = %s, ef_construction = %d)" % (self._m, self._ef_construction)
                )
            elif self._metric == "euclidean":
                cur.execute("CREATE INDEX pgv_idx ON items USING hnsw (embedding vector_l2_ops) WITH (m = %s, ef_construction = %d)" % (self._m, self._ef_construction))
            else:
                raise RuntimeError(f"unknown metric {self._metric}")
            # reset back to the default value after index creation
            cur.execute("SET maintenance_work_mem = '2GB'")
            print("Prewarming index...")
            cur.execute("SELECT pg_prewarm('pgv_idx', 'buffer')")
            print("Index prewarming done!")
        self._cur = cur

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw.ef_search = %d" % ef_search)
        print("SET hnsw.ef_search = %d" % ef_search)
        # TODO set based on available memory
        self._cur.execute("SET work_mem = '8GB'")
        # disable parallel query execution
        self._cur.execute("SET max_parallel_workers_per_gather = 0")
        self._cur.execute("SET enable_seqscan=0")

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def __str__(self):
        return f"PGVectorHNSW(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
