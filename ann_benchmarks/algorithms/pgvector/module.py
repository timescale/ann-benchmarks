import subprocess
import sys
import pgvector.psycopg
import psycopg
from ..base.module import BaseANN
from psycopg_pool import ConnectionPool

class PGVector(BaseANN):
    def __init__(self, metric, lists):
        self._metric = metric
        self._lists = lists
        self._pool = None
        self._probes = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def start_pool(self):
        def configure(conn):
            pgvector.psycopg.register_vector(conn)
            if self._probes is not None:
                conn.execute("SET ivfflat.probes = %d" % self._probes)
                print("SET ivfflat.probes = %d" % self._probes)
                conn.execute("SET work_mem = '8GB'")
                # disable parallel query execution
                conn.execute("SET max_parallel_workers_per_gather = 0")
                conn.execute("SET enable_seqscan=0")
            conn.commit()

        self._pool = ConnectionPool("postgresql://ann:ann@localhost/ann",
                                    min_size=1, max_size=16, configure=configure)

    def fit(self, X):
        #subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        self.start_pool()
        with self._pool.connection() as conn:
            cur = conn.cursor()
            cur.execute("select count(*) from pg_class where relname = 'items'")
            table_count = cur.fetchone()[0]

            if table_count == 0:
                cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
                cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
                conn.commit()
                print("copying data...")
                with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
                    for i, embedding in enumerate(X):
                        copy.write_row((i, embedding))
                cur.execute("COMMIT")        

            cur.execute("drop index if exists pgv_idx")
            cur.execute("select count(*) from pg_indexes where indexname = 'pgv_idx'")
            index_count = cur.fetchone()[0]

            if index_count == 0:                    
                print("Creating Index (lists = %d)" % self._lists)
                cur.execute("SET maintenance_work_mem = '16GB'")
                if self._metric == "angular":
                    cur.execute(
                        "CREATE INDEX pgv_idx ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = %d)" % self._lists
                    )
                elif self._metric == "euclidean":
                    cur.execute("CREATE INDEX pgv_idx ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = %d)" % self._lists)
                else:
                    raise RuntimeError(f"unknown metric {self._metric}")
                # reset back to the default value after index creation
                cur.execute("SET maintenance_work_mem = '2GB'")
                conn.commit()
            print("Prewarming index...")
            cur.execute("SELECT pg_prewarm('pgv_idx', 'buffer')")
            print("Index prewarming done!")

    def set_query_arguments(self, probes):
        self._probes = probes
        #close and restart the pool to apply the new settings
        self._pool.close()
        self._pool = None
        self.start_pool()

    def get_memory_usage(self):
        if self._pool is None:
            return 0
        with self._pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT pg_relation_size('pgv_idx')")
                return cursor.fetchone()[0] / 1024

    def query(self, v, n):
        with self._pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(self._query, (v, n), binary=True, prepare=True)
                return [id for id, in cursor.fetchall()]

    def __str__(self):
        return f"PGVector(lists={self._lists}, probes={self._probes})"
