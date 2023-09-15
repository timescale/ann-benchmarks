import subprocess
import sys
import psycopg
from ..base.module import BaseANN
from   psycopg_pool import ConnectionPool

class PGEmbeddingHNSW(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param["M"]
        self._ef_construction = method_param["efConstruction"]
        self._pool = None
        self._ef = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s::real[] LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s::real[] LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def start_pool(self):
        def configure(conn):
            #pgvector.psycopg.register_vector(conn)
            if self._ef is not None:
                conn.execute("ALTER INDEX items_embedding_idx  SET (efsearch=%d)" % self._ef)
                print("ALTER INDEX items_embedding_idx SET (efsearch=%d)" % self._ef)
                # disable parallel query execution
                conn.execute("SET work_mem = '8GB'")
                # disable parallel query execution
                conn.execute("SET max_parallel_workers_per_gather = 0")
                conn.execute("SET enable_seqscan=0")
            conn.commit()

        self._pool = ConnectionPool("postgresql://ann:ann@localhost/ann",
                                    min_size=1, max_size=16, configure=configure)

    def fit(self, X):
        self.start_pool()
        with self._pool.connection() as conn:
            cur = conn.cursor()
            cur.execute("select count(*) from pg_class where relname = 'items'")
            table_count = cur.fetchone()[0]

            if table_count == 0:
                cur.execute("CREATE TABLE items (id int, embedding real[])")
                cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
                conn.commit()
                print("copying data...")
                with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
                    for i, embedding in enumerate(X):
                        copy.write_row((i, embedding.tolist()))
                cur.execute("COMMIT")

            cur.execute("drop index if exists items_embedding_idx")
            cur.execute("select count(*) from pg_indexes where indexname = 'items_embedding_idx'")
            index_count = cur.fetchone()[0]

            if index_count == 0:
                cur.execute("SET maintenance_work_mem = '16GB'")
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
                # reset back to the default value after index creation
                cur.execute("SET maintenance_work_mem = '2GB'")
                conn.commit()
            print("prewarming index...")
            cur.execute("SELECT pg_prewarm('items_embedding_idx', 'buffer')")
            print("Index prewarming done!")

    def set_query_arguments(self, ef):
        self._ef = ef
        #close and restart the pool to apply the new settings
        self._pool.close()
        self._pool = None
        self.start_pool()
 
    def get_memory_usage(self):
        if self._pool is None:
            return 0
        with self._pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT pg_relation_size('items_embedding_idx')")
                return cursor.fetchone()[0] / 1024

    def query(self, v, n):
        with self._pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(self._query, (v.tolist(), n), binary=True, prepare=True)
                return [id for id, in cursor.fetchall()]

    def __str__(self):
        return f"PGEmbedding(m={self._m}, probes={self._ef_construction}, query_ef_search={self._ef})"
