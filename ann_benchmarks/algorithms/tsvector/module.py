from ..base.module import BaseANN
import numpy
import concurrent.futures
from typing import Optional
import psutil
import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from datetime import datetime, timezone, timedelta

MAX_CONNECTIONS = 32

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
        self._pool = ConnectionPool(self._connection_str, min_size=1, max_size=MAX_CONNECTIONS, configure=configure)

    def does_table_exist(self, conn: psycopg.Connection) -> bool:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_class WHERE relname = 'items'")
            table_count = cur.fetchone()[0]
            return table_count > 0
        
    def create_table(self, conn: psycopg.Connection, dimensions: int) -> None:
        with conn.cursor() as cur:
            print("creating table...")
            cur.execute(f"CREATE TABLE public.items (id int, t timestamptz, embedding vector({dimensions}))")
            cur.execute("ALTER TABLE public.items ALTER COLUMN embedding SET STORAGE PLAIN")
            cur.execute("SELECT create_hypertable('public.items'::regclass, 't'::name, chunk_time_interval=>'1d'::interval)")
            conn.commit()

    def load_table(self, conn: psycopg.Connection, X: numpy.array) -> None:
        with conn.cursor() as cur:
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
    
    def list_chunks(self, conn: psycopg.Connection) -> list[str]:
        chunks = []
        with conn.cursor() as cur:
            cur.execute("""
                select format('%I.%I', chunk_schema, chunk_name)
                from timescaledb_information.chunks
                where hypertable_schema = 'public'
                and hypertable_name = 'items'
                """)
            for row in cur:
                chunks.append(row[0])
        return chunks
    
    def index_chunk(self, chunk: str) -> Optional[Exception]:
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    if self._pq_vector_length < 1:
                        cur.execute(f"""CREATE INDEX ON ONLY {chunk} USING tsv (embedding) 
                            WITH (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size}, max_alpha={self._max_alpha})""",
                        )
                    else:
                        cur.execute(f"""CREATE INDEX ON ONLY {chunk} USING tsv (embedding) 
                            WITH (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size}, 
                            max_alpha = {self._max_alpha}, use_pq=true, pq_vector_length = {self._pq_vector_length})"""
                        )
                    conn.commit()
        except Exception as x:
            return x
        return None

    def index_chunks(self, chunks: list[str]) -> None:
        threads = min(MAX_CONNECTIONS, len(chunks))
        print(f"creating indexes using {threads} threads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_chunk = {executor.submit(self.index_chunk, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    x = future.result()
                except Exception as x2:
                    print(f"creating index on chunk {chunk} hit an exception: {x2}")
                else:
                    if x is not None:
                        print(f"creating index on chunk {chunk} hit an exception: {x}")
                    else:
                        print(f"created index on {chunk}")
        print("finished creating indexes")

    def fit(self, X: numpy.array) -> None:
        self.start_pool()
        chunks: list[str] = None
        with self._pool.connection() as conn:
            if self.does_table_exist(conn):
                return
            self.create_table(conn, int(X.shape[1]))
            self.load_table(conn, X)
            chunks = self.list_chunks(conn)
        self.index_chunks(chunks)

    def set_query_arguments(self, query_search_list_size):
        self._query_search_list_size = query_search_list_size
        #close and restart the pool to apply the new settings
        self._pool.close()
        self._pool = None
        self.start_pool()

    def get_memory_usage(self) -> Optional[float]:
        return psutil.Process().memory_info().rss / 1024
        #if self._pool is None:
        #    return 0
        #with self._pool.connection() as conn:
        #    with conn.cursor() as cursor:
        #        cursor.execute("SELECT pg_relation_size('idx_tsv')")
        #        return cursor.fetchone()[0] / 1024

    def query(self, q: numpy.array, n: int) -> numpy.array:
        with self._pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(self._query, (q, n), binary=True, prepare=True)
                return numpy.array([id for id, in cursor.fetchall()])

    def batch_query(self, X: numpy.array, n: int) -> None:
        threads = min(MAX_CONNECTIONS, X.size)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(self.query, q, n) for q in X]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as x2:
                    print(f"exception getting batch results: {x2}")
        self.res = numpy.array(results)

    def get_batch_results(self) -> numpy.array:
        return self.res

    def __str__(self):
        return f"TSVector(num_neighbors={self._num_neighbors}, search_list_size={self._search_list_size}, max_alpha={self._max_alpha}, pq_vector_length={self._pq_vector_length}, query_search_list_size={self._query_search_list_size})"
