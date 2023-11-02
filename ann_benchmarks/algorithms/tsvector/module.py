from ..base.module import BaseANN
import numpy
import concurrent.futures
from typing import Optional
import psutil
import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from datetime import datetime, timezone, timedelta
import subprocess
import os
import sys
import shutil


MAX_DB_CONNECTIONS = 32
MAX_CREATE_INDEX_THREADS = 32
MAX_BATCH_QUERY_THREADS = 8
EMBEDDINGS_PER_COPY_BATCH = 5_000 # how many rows per COPY statement
EMBEDDINGS_PER_CHUNK = 100_000 # how many rows per hypertable chunk
START_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc) # minimum time used for time column
CHUNK_TIME_STEP = timedelta(days=1) # how much to increment the time column by for each chunk
CHUNK_TIME_INTERVAL = "'1d'::interval"

assert(EMBEDDINGS_PER_COPY_BATCH <= EMBEDDINGS_PER_CHUNK)


class TSVector(BaseANN):
    def __init__(self, metric: str, connection_str: str, num_neighbors: int, search_list_size: int,
                max_alpha: float, pq_vector_length: int, query_search_list_size: int):
        self._metric: str = metric
        self._connection_str: str = connection_str
        self._num_neighbors: int = num_neighbors
        self._search_list_size: int = search_list_size
        self._max_alpha: float = max_alpha
        self._pq_vector_length: int = pq_vector_length
        self._query_search_list_size: Optional[int] = query_search_list_size if query_search_list_size > 0 else None
        self._pool : ConnectionPool = None

        if metric == "euclidean":
            self._query: str = "SELECT id FROM public.items ORDER BY embedding <=> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")
        print(f"query: {self._query}")

    def start_pool(self):
        def configure(conn):
            register_vector(conn)
            if self._query_search_list_size is not None:
                conn.execute("SET tsv.query_search_list_size = %d" % self._query_search_list_size)
                print("SET tsv.query_search_list_size = %d" % self._query_search_list_size)
                # conn.execute("SET work_mem = '8GB'")
                # disable parallel query execution
                conn.execute("SET max_parallel_workers_per_gather = 0")
                conn.execute("SET enable_seqscan=0")
            conn.commit()
        self._pool = ConnectionPool(self._connection_str, min_size=1, max_size=MAX_DB_CONNECTIONS, configure=configure)

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
            cur.execute(f"SELECT create_hypertable('public.items'::regclass, 't'::name, chunk_time_interval=>{CHUNK_TIME_INTERVAL})")
            conn.commit()

    def load_table_serial(self, X: numpy.array) -> None:
        batches: list[numpy.array] = None
        if X.shape[0] < EMBEDDINGS_PER_COPY_BATCH:
            batches = [X]
        else:
            splits = [x for x in range(0, X.shape[0], EMBEDDINGS_PER_COPY_BATCH)][1:]
            batches = numpy.split(X, splits)
        print(f"copying {X.shape[0]} rows into table using {len(batches)} batches...")
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                i = -1
                d = START_TIME - CHUNK_TIME_STEP
                for b, batch in enumerate(batches):
                    print(f"copying batch number {b} of {batch.shape[0]} rows into chunk {d}")
                    with cur.copy("COPY public.items (id, t, embedding) FROM STDIN") as copy:
                        for v in batch:
                            i += 1
                            if i % EMBEDDINGS_PER_CHUNK == 0:
                                d = d + CHUNK_TIME_STEP
                            copy.write_row((i, d, v))
                        conn.commit()

    def load_table_parallel(self, X: numpy.array) -> None:
        cmd = f"""timescaledb-parallel-copy -connection '{self._connection_str}' -columns 'id,t,embedding' -schema public -table items -workers 32 -log-batches"""
        p = subprocess.Popen(args=cmd, stdin=subprocess.PIPE, stdout=sys.stdout, stderr=subprocess.STDOUT, text=True, shell=True, env=os.environ)
        i = -1
        d = START_TIME - CHUNK_TIME_STEP
        for v in X:
            i += 1
            if i > 0 and i % EMBEDDINGS_PER_COPY_BATCH == 0:
                p.stdin.flush()
            if i % EMBEDDINGS_PER_CHUNK == 0:
                d = d + CHUNK_TIME_STEP
                v2 = '"[' + ",".join([f"{j}" for j in v]) + ']"'
            p.stdin.write(f"""{i},"{d.isoformat()}",{v2}\n""")
        p.stdin.flush()
        p.stdin.close()
        retcode = p.wait()
        if retcode != 0:
            raise subprocess.CalledProcessError(returncode=retcode, cmd=cmd)
        print("finished loading the table")

    def load_table(self, X: numpy.array) -> None:
        if shutil.which("timescaledb-parallel-copy") is None:
            return self.load_table_serial(X)
        else:
            return self.load_table_parallel(X)

    def list_chunks(self, conn: psycopg.Connection) -> list[str]:
        with conn.cursor() as cur:
            cur.execute("""
                select format('%I.%I', chunk_schema, chunk_name)
                from timescaledb_information.chunks k
                where hypertable_schema = 'public'
                and hypertable_name = 'items'
                and not exists
                (
                    select 1
                    from pg_catalog.pg_indexes i
                    where k.chunk_schema = i.schemaname
                    and k.chunk_name = i.tablename
                    and i.indexname like '%_embedding_%'
                )
                """)
            return [row[0] for row in cur]

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
        if len(chunks) == 0:
            return
        threads = min(MAX_CREATE_INDEX_THREADS, len(chunks))
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
        # have to create the extensions before starting the connection pool
        with psycopg.connect(self._connection_str) as conn:
            with conn.cursor() as cur:
                cur.execute("create extension if not exists timescaledb")
                cur.execute("create extension if not exists timescale_vector cascade")
        self.start_pool()
        table_exists:bool = False
        with self._pool.connection() as conn:
            table_exists = self.does_table_exist(conn)
            if not table_exists:
                self.create_table(conn, int(X.shape[1]))
        if not table_exists:
            self.load_table(X)
        chunks: list[str] = None
        with self._pool.connection() as conn:
            chunks = self.list_chunks(conn)
        if len(chunks) > 0:
            self.index_chunks(chunks)

    def set_query_arguments(self, query_search_list_size):
        self._query_search_list_size = query_search_list_size
        #close and restart the pool to apply the new settings
        self._pool.close()
        self._pool = None
        self.start_pool()

    def get_memory_usage(self) -> Optional[float]:
        return psutil.Process().memory_info().rss / 1024

    def query(self, q: numpy.array, n: int) -> numpy.array:
        with self._pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(self._query, (q, n), binary=True, prepare=True)
                return numpy.array([id for id, in cursor.fetchall()])

    def batch_query(self, X: numpy.array, n: int) -> None:
        threads = min(MAX_BATCH_QUERY_THREADS, X.size)
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
