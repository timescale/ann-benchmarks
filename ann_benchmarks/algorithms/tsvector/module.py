from ..base.module import BaseANN
import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector


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
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        #if metric == "angular":
        #    self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        #elif metric == "euclidean":
        #    self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
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
            cur.execute("select count(*) from pg_class where relname = 'items'")
            table_count = cur.fetchone()[0]

            if table_count == 0:
                print("creating table...")
                cur.execute(f"CREATE TABLE items (id int, embedding vector({int(X.shape[1])}))")
                cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
                conn.commit()
                print("copying data...")
                with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
                    for i, embedding in enumerate(X):
                        copy.write_row((i, embedding))
                conn.commit()

            cur.execute("drop index if exists idx_tsv")
            cur.execute("SET maintenance_work_mem = '16GB'")
            if self._metric != "angular" and self._metric != "euclidean":
                raise RuntimeError(f"unknown metric {self._metric}")
            if self._pq_vector_length < 1:
                cur.execute(f"""CREATE INDEX idx_tsv ON items USING tsv (embedding) 
                    WITH (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size}, max_alpha={self._max_alpha})""",
                )
            else:
                cur.execute(f"""CREATE INDEX idx_tsv ON items USING tsv (embedding) 
                    WITH (num_neighbors = {self._num_neighbors}, search_list_size = {self._search_list_size}, 
                    max_alpha = {self._max_alpha}, use_pq=true, pq_vector_length = {self._pq_vector_length})"""
                )
            # reset back to the default value after index creation
            cur.execute("SET maintenance_work_mem = '2GB'")
            conn.commit()
            #print("Prewarming index...")
            #cur.execute("SELECT pg_prewarm('idx_tsv', 'buffer')")
            #print("Index prewarming done!")

    def set_query_arguments(self, query_search_list_size):
        self._query_search_list_size = query_search_list_size
        #close and restart the pool to apply the new settings
        self._pool.close()
        self._pool = None
        self.start_pool()

    def get_memory_usage(self):
        if self._pool is None:
            return 0
        with self._pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT pg_relation_size('idx_tsv')")
                return cursor.fetchone()[0] / 1024

    def query(self, v, n):
        with self._pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(self._query, (v, n), binary=True, prepare=True)
                return [id for id, in cursor.fetchall()]

    def __str__(self):
        return f"TSVector(num_neighbors={self._num_neighbors}, search_list_size={self._search_list_size}, max_alpha={self._max_alpha}, pq_vector_length={self._pq_vector_length}, query_search_list_size={self._query_search_list_size})"
