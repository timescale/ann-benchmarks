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
from time import perf_counter

LOAD_PARALLEL = False
EMBEDDINGS_PER_CHUNK = 1_000_000  # how many rows per hypertable chunk
QUERY = """select id from public.items order by embedding <=> %s limit %s"""
# QUERY = """with x as materialized (select id, embedding <=> %s as distance from public.items order by 2 limit 100) select id from x order by distance limit %s"""

CONNECTION_SETTINGS = [
    "set work_mem = '2GB';",
    "set maintenance_work_mem = '8GB';"
    "set max_parallel_workers_per_gather = 0;",
    "set enable_seqscan=0;",
    "set jit = 'off';",
]

MAX_DB_CONNECTIONS = 16
MAX_CREATE_INDEX_THREADS = 16
MAX_BATCH_QUERY_THREADS = 16
EMBEDDINGS_PER_COPY_BATCH = 5_000  # how many rows per COPY statement
# minimum time used for time column
START_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc)
# how much to increment the time column by for each chunk
CHUNK_TIME_STEP = timedelta(days=1)
CHUNK_TIME_INTERVAL = "'1d'::interval"
STORAGE_LAYOUT = "memory_optimized"
PREWARM = True

assert (EMBEDDINGS_PER_COPY_BATCH <= EMBEDDINGS_PER_CHUNK)


class TSVector(BaseANN):
    def __init__(self, metric: str, connection_str: str, num_neighbors: int, search_list_size: int,
                 max_alpha: float, use_bq: int, pq_vector_length: int, num_bits_per_dimension: int):
        self._metric: str = metric
        self._connection_str: str = connection_str
        self._num_neighbors: int = num_neighbors
        self._search_list_size: int = search_list_size
        self._max_alpha: float = max_alpha
        self._use_bq: bool = (use_bq == 1)
        self._pq_vector_length: int = pq_vector_length
        self._num_bits_per_dimension: int = num_bits_per_dimension
        self._query_search_list_size: Optional[int] = None
        self._query_rescore: Optional[int] = None
        self._query_shared_buffers_hit = 0
        self._query_shared_buffers_read = 0
        self._pool: ConnectionPool = None
        if metric == "angular":
            self._query: str = QUERY
        else:
            raise RuntimeError(f"unknown metric {metric}")
        print(f"query: {self._query}")

    def create_log_table(self, cur: psycopg.Cursor) -> None:
        cur.execute("create table if not exists public.log (id bigint not null generated by default as identity primary key, name text, start timestamptz not null default clock_timestamp(), stop timestamptz)")

    def log_start(self, conn: psycopg.Connection, name: str) -> int:
        with conn.cursor() as cur:
            cur.execute(
                "insert into public.log (name) values (%s) returning id", (name, ))
            return int(cur.fetchone()[0])

    def log_stop(self, conn: psycopg.Connection, id: int) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "update public.log set stop = clock_timestamp() where id = %s", (id,))

    def start_pool(self):
        def configure(conn):
            register_vector(conn)
            if self._query_search_list_size is not None:
                conn.execute("set tsv.query_search_list_size = %d" %
                             self._query_search_list_size)
                print("set tsv.query_search_list_size = %d" %
                      self._query_search_list_size)
            if self._query_rescore is not None:
                conn.execute("set tsv.query_rescore = %d" %
                             self._query_rescore)
                print("set tsv.query_rescore = %d" % self._query_rescore)
            for setting in CONNECTION_SETTINGS:
                conn.execute(setting)
            conn.commit()
        self._pool = ConnectionPool(
            self._connection_str, min_size=1, max_size=MAX_DB_CONNECTIONS, configure=configure)

    def does_table_exist(self, conn: psycopg.Connection) -> bool:
        table_count = 0
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from pg_class where relname = 'items'")
            table_count = cur.fetchone()[0]
        return table_count > 0

    def shared_buffers(self, conn: psycopg.Connection):
        shared_buffers_hit = 0
        shared_buffers_read = 0
        with conn.cursor() as cur:
            sql_query = QUERY % ("$1", "$2")
            cur.execute(f"""
                select
                    shared_blks_hit, shared_blks_read
                from pg_stat_statements
                where queryid = (select queryid
                from pg_stat_statements
                where userid = (select oid from pg_roles where rolname = current_role)
                and query like '{sql_query}'
                );""")
            res = cur.fetchone()
            if res is not None:
                shared_buffers_hit = res[0]
                shared_buffers_read = res[1]
        return shared_buffers_hit, shared_buffers_read

    def create_table(self, conn: psycopg.Connection, dimensions: int) -> None:
        with conn.cursor() as cur:
            print("creating table...")
            cur.execute(
                f"create table public.items (id int, t timestamptz, embedding vector({dimensions}))")
            cur.execute(
                "alter table public.items alter column embedding set storage plain")
            cur.execute(
                f"select create_hypertable('public.items'::regclass, 't'::name, chunk_time_interval=>{CHUNK_TIME_INTERVAL})")
            conn.commit()

    def load_table_binary(self, X: numpy.array) -> None:
        batches: list[numpy.array] = None
        if X.shape[0] < EMBEDDINGS_PER_COPY_BATCH:
            batches = [X]
        else:
            splits = [x for x in range(
                0, X.shape[0], EMBEDDINGS_PER_COPY_BATCH)][1:]
            batches = numpy.split(X, splits)
        print(
            f"copying {X.shape[0]} rows into table using {len(batches)} batches...")
        with self._pool.connection() as con:
            with con.cursor(binary=True) as cur:
                i = -1
                d = START_TIME - CHUNK_TIME_STEP
                for b, batch in enumerate(batches):
                    print(
                        f"copying batch number {b} of {batch.shape[0]} rows into chunk {d}")
                    with cur.copy("copy public.items (id, t, embedding) from stdin (format binary)") as cpy:
                        cpy.set_types(['integer', 'timestamptz', 'vector'])
                        for v in batch:
                            i += 1
                            if i % EMBEDDINGS_PER_CHUNK == 0:
                                d = d + CHUNK_TIME_STEP
                            cpy.write_row((i, d, v))
                    con.commit()

    def load_table_serial(self, X: numpy.array) -> None:
        batches: list[numpy.array] = None
        if X.shape[0] < EMBEDDINGS_PER_COPY_BATCH:
            batches = [X]
        else:
            splits = [x for x in range(
                0, X.shape[0], EMBEDDINGS_PER_COPY_BATCH)][1:]
            batches = numpy.split(X, splits)
        print(
            f"copying {X.shape[0]} rows into table using {len(batches)} batches...")
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                i = -1
                d = START_TIME - CHUNK_TIME_STEP
                for b, batch in enumerate(batches):
                    print(
                        f"copying batch number {b} of {batch.shape[0]} rows into chunk {d}")
                    with cur.copy("copy public.items (id, t, embedding) from stdin") as copy:
                        for v in batch:
                            i += 1
                            if i % EMBEDDINGS_PER_CHUNK == 0:
                                d = d + CHUNK_TIME_STEP
                            copy.write_row((i, d, v))
                    conn.commit()

    def load_table_parallel(self, X: numpy.array) -> None:
        print(f"total dataset: {X.shape[0]}")
        i = -1
        d = START_TIME - CHUNK_TIME_STEP
        cmd = f"""timescaledb-parallel-copy -connection '{self._connection_str}' -workers {MAX_DB_CONNECTIONS} -batch-size {EMBEDDINGS_PER_COPY_BATCH} -columns 'id,t,embedding' -schema public -table items -log-batches"""
        p = subprocess.Popen(args=cmd, stdin=subprocess.PIPE, stdout=sys.stdout,
                             stderr=subprocess.STDOUT, text=True, shell=True, env=os.environ)
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
        id = 0
        with self._pool.connection() as conn:
            id = self.log_start(conn, "loading table")
        if LOAD_PARALLEL and shutil.which("timescaledb-parallel-copy") is not None:
            self.load_table_parallel(X)
        else:
            self.load_table_binary(X)
        with self._pool.connection() as conn:
            self.log_stop(conn, id)

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
                order by k.range_start
                """)
            return [row[0] for row in cur]

    def index_table(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if self._use_bq:
                    cur.execute(f"""create index on only public.items using tsv (embedding)
                        with (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size}, max_alpha={self._max_alpha},
                          num_bits_per_dimension={self._num_bits_per_dimension}, storage_layout='{STORAGE_LAYOUT}')"""
                                )
                elif self._pq_vector_length < 1:
                    cur.execute(f"""create index on only public.items using tsv (embedding)
                        with (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size}, max_alpha={self._max_alpha})""",
                                )
                else:
                    cur.execute(f"""create index on only public.items using tsv (embedding)
                        with (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size},
                        max_alpha = {self._max_alpha}, use_pq=true, pq_vector_length = {self._pq_vector_length})"""
                                )
                conn.commit()

    def index_chunk(self, chunk: str) -> Optional[Exception]:
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    if self._use_bq:
                        cur.execute(f"""create index on only {chunk} using tsv (embedding)
                            with (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size}, max_alpha={self._max_alpha},
                            num_bits_per_dimension={self._num_bits_per_dimension}, storage_layout='{STORAGE_LAYOUT}')"""
                                    )
                    elif self._pq_vector_length < 1:
                        cur.execute(f"""create index on only {chunk} using tsv (embedding)
                            with (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size}, max_alpha={self._max_alpha})""",
                                    )
                    else:
                        cur.execute(f"""create index on only {chunk} using tsv (embedding)
                            with (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size},
                            max_alpha = {self._max_alpha}, use_pq=true, pq_vector_length = {self._pq_vector_length})"""
                                    )
                    conn.commit()
        except Exception as x:
            return x
        return None

    def index_chunks(self, chunks: list[str]) -> None:
        id = 0
        with self._pool.connection() as conn:
            id = self.log_start(conn, "indexing")
        if len(chunks) == 0:
            return
        threads = min(MAX_CREATE_INDEX_THREADS, len(chunks))
        print(f"creating indexes using {threads} threads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_chunk = {executor.submit(
                self.index_chunk, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    x = future.result()
                except Exception as x2:
                    print(
                        f"creating index on chunk {chunk} hit an exception: {x2}")
                else:
                    if x is not None:
                        print(
                            f"creating index on chunk {chunk} hit an exception: {x}")
                    else:
                        print(f"created index on {chunk}")
        print("finished creating indexes")
        with self._pool.connection() as conn:
            self.log_stop(conn, id)

    def prewarm_heap(self, conn: psycopg.Connection) -> None:
        if PREWARM:
            with conn.cursor() as cur:
                cur.execute(
                    "select format($$%I.%I$$, chunk_schema, chunk_name) from timescaledb_information.chunks k where hypertable_name = 'items'")
                chunks = [row[0] for row in cur]
                for chunk in chunks:
                    print(f"prewarming chunk heap {chunk}")
                    cur.execute(
                        f"select pg_prewarm('{chunk}'::regclass, mode=>'buffer')")
                    cur.fetchall()

    def prewarm_index(self, conn: psycopg.Connection) -> None:
        if PREWARM:
            with conn.cursor() as cur:
                cur.execute("""
                    select format($$%I.%I$$, x.schemaname, x.indexname)
                    from timescaledb_information.chunks k
                    inner join pg_catalog.pg_indexes x on (k.chunk_schema = x.schemaname and k.chunk_name = x.tablename)
                    where x.indexname ilike '%_embedding_%'
                    and k.hypertable_name = 'items'""")
                chunks = [row[0] for row in cur]
                for chunk_index in chunks:
                    print(f"prewarming chunk index {chunk_index}")
                    cur.execute(
                        f"select pg_prewarm('{chunk_index}'::regclass, mode=>'buffer')")
                    cur.fetchall()

    def fit(self, X: numpy.array) -> None:
        # have to create the extensions before starting the connection pool
        with psycopg.connect(self._connection_str) as conn:
            with conn.cursor() as cur:
                cur.execute("create extension if not exists timescaledb")
                cur.execute(
                    "create extension if not exists timescale_vector cascade")
                self.create_log_table(conn)
        self.start_pool()
        table_exists: bool = False
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

    def set_query_arguments(self, query_search_list_size, query_rescore):
        self._query_search_list_size = query_search_list_size
        self._query_rescore = query_rescore
        # close and restart the pool to apply the new settings
        self._pool.close()
        self._pool = None
        self.start_pool()
        with self._pool.connection() as conn:
            self.prewarm_heap(conn)
            self.prewarm_index(conn)

    def get_memory_usage(self) -> Optional[float]:
        return psutil.Process().memory_info().rss / 1024

    def query(self, q: numpy.array, n: int) -> tuple[numpy.array, float]:
        start = perf_counter()
        with self._pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(self._query, (q, n), binary=True, prepare=True)
                result = numpy.array([id for id, in cursor.fetchall()])
                elapsed = perf_counter() - start
                return result, elapsed

    def batch_query(self, X: numpy.array, n: int) -> None:
        threads = min(MAX_BATCH_QUERY_THREADS, X.size)

        with self._pool.connection() as conn:
            shared_buffers_start_hit, shared_buffers_start_read = self.shared_buffers(
                conn)

        results = numpy.empty((X.shape[0], n), dtype=int)
        latencies = numpy.empty(X.shape[0], dtype=float)
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(
                self.query, q, n): i for i, q in enumerate(X)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    result, latency = future.result()
                    results[i] = result
                    latencies[i] = latency
                except Exception as x2:
                    print(f"exception getting batch results: {x2}")
        self.results = results
        self.latencies = latencies

        with self._pool.connection() as conn:
            shared_buffers_end_hit, shared_buffers_end_read = self.shared_buffers(
                conn)

        self._query_shared_buffers_hit = shared_buffers_end_hit - shared_buffers_start_hit
        self._query_shared_buffers_read = shared_buffers_end_read - shared_buffers_start_read

    def get_additional(self):
        return {"shared_buffers": self._query_shared_buffers_hit + self._query_shared_buffers_read,
                "shared_buffers_hit": self._query_shared_buffers_hit,
                "shared_buffers_read": self._query_shared_buffers_read
                }

    def get_batch_results(self) -> numpy.array:
        return self.results

    def get_batch_latencies(self) -> numpy.array:
        return self.latencies

    def __str__(self):
        return f"algorithm=TSVector num_neighbors={self._num_neighbors} search_list_size={self._search_list_size} max_alpha={self._max_alpha} num_bits_per_dimension={self._num_bits_per_dimension} use_bq={self._use_bq} pq_vector_length={self._pq_vector_length} query_search_list_size={self._query_search_list_size} query_rescore={self._query_rescore}"
