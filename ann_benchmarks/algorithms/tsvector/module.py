from ..base.module import BaseANN
import numpy
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Optional
import psutil
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from datetime import datetime, timezone, timedelta

class TSVector(BaseANN):
    def __init__(self, metric, connection_str, num_neighbors, search_list_size, max_alpha, pq_vector_length, query_search_list_size):
        self._metric = metric
        self._connection_str = connection_str
        self._num_neighbors = num_neighbors
        self._search_list_size = search_list_size
        self._max_alpha = max_alpha
        self._pq_vector_length = pq_vector_length
        self._query_search_list_size = query_search_list_size if query_search_list_size > 0 else None
        self._pool = None

        if metric == "euclidean":
            self._query = "SELECT id FROM public.items ORDER BY embedding <=> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")
        print(f"query: {self._query}")

    def start_pool(self):
        def configure(conn):
            register_vector(conn)
            if self._query_search_list_size is not None:
                conn.execute("SET tsv.query_search_list_size = %d" % self._query_search_list_size)
                print("SET tsv.query_search_list_size = %d" % self._query_search_list_size)
                conn.execute("SET work_mem = '8GB'")
                # disable parallel query execution
                conn.execute("SET max_parallel_workers_per_gather = 0")
                conn.execute("SET enable_seqscan=0")
            conn.commit()
        self._pool = ConnectionPool(self._connection_str, min_size=1, max_size=16, configure=configure)

    def fit(self, X):
        self.start_pool()
        with self._pool.connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM pg_class WHERE relname = 'items'")
            table_count = cur.fetchone()[0]
            if table_count == 0:
                print("creating table...")
                cur.execute(f"CREATE TABLE public.items (id int, t timestamptz, embedding vector({int(X.shape[1])}))")
                cur.execute("ALTER TABLE public.items ALTER COLUMN embedding SET STORAGE PLAIN")
                cur.execute("SELECT create_hypertable('public.items'::regclass, 't'::name, chunk_time_interval=>'1d'::interval)")
                conn.commit()
                i = 0
                d = datetime(2000, 1, 1, tzinfo=timezone.utc)
                e = enumerate(X)
                print(f"copying {len(X)} rows into table...")
                while i < len(X):
                    print(f"copying data into chunk: {d}")
                    with cur.copy("COPY public.items (id, t, embedding) FROM STDIN") as copy:
                        while i < len(X):
                            _, v = next(e)
                            copy.write_row((i, d, v))
                            i += 1
                            if i % 100_000 == 0:
                                d = d + timedelta(days=1.0)
                                break
                    print(f"i = {i} committing...")
                    conn.commit()

            cur.execute("select count(*) from pg_indexes where schemaname = 'public' and tablename = 'items' and indexname = 'idx_tsv'")
            index_count = cur.fetchone()[0]
            if index_count == 0:
                print("creating index...")
                cur.execute("DROP INDEX if exists idx_tsv")
                cur.execute("SET maintenance_work_mem = '16GB'")
                if self._metric != "euclidean":
                    raise RuntimeError(f"unknown metric {self._metric}")
                if self._pq_vector_length < 1:
                    cur.execute(f"""CREATE INDEX idx_tsv ON public.items USING tsv (embedding) 
                        WITH (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size}, max_alpha={self._max_alpha})""",
                    )
                else:
                    cur.execute(f"""CREATE INDEX idx_tsv ON public.items USING tsv (embedding) 
                        WITH (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size}, 
                        max_alpha = {self._max_alpha}, use_pq=true, pq_vector_length = {self._pq_vector_length})"""
                    )
                # reset back to the default value after index creation
                cur.execute("SET maintenance_work_mem = '2GB'")
                conn.commit()
                print("index created")

    def set_query_arguments(self, query_search_list_size):
        self._query_search_list_size = query_search_list_size
        #close and restart the pool to apply the new settings
        self._pool.close()
        self._pool = None
        self.start_pool()

    def get_memory_usage(self):
        return psutil.Process().memory_info().rss / 1024
        #if self._pool is None:
        #    return 0
        #with self._pool.connection() as conn:
        #    with conn.cursor() as cursor:
        #        cursor.execute("SELECT pg_relation_size('idx_tsv')")
        #        return cursor.fetchone()[0] / 1024

    def query(self, v, n):
        with self._pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(self._query, (v, n), binary=True, prepare=True)
                return [id for id, in cursor.fetchall()]

    def batch_query(self, X: numpy.array, n: int) -> None:
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self) -> numpy.array:
        return self.res

    def __str__(self):
        return f"TSVector(num_neighbors={self._num_neighbors}, search_list_size={self._search_list_size}, max_alpha={self._max_alpha}, pq_vector_length={self._pq_vector_length}, query_search_list_size={self._query_search_list_size})"
