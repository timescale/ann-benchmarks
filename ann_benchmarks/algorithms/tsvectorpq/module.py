import subprocess
import sys
import pgvector.psycopg
import psycopg
from ..base.module import BaseANN

class TSVectorpq(BaseANN):
    def __init__(self, metric, num_neighbors, search_list_size, max_alpha, pq_vector_length):
        self._metric = metric
        self._num_neighbors = num_neighbors
        self._search_list_size = search_list_size
        self._max_alpha = max_alpha
        self._pq_vector_length = pq_vector_length
        self._cur = None

        if metric == "angular":
            self._query = "with raw as (SELECT id, embedding FROM items order by embedding <=> %s limit %s) select id from raw order by embedding <=> %s limit %s"
        else:
            if metric == "euclidean":
                self._query = "with raw as (SELECT id, embedding FROM items order by embedding <-> %s limit %s) select id from raw order by embedding <-> %s limit %s"
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
            cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
            cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
            print("copying data...")
            with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
                for i, embedding in enumerate(X):
                    copy.write_row((i, embedding))
            conn.commit();

        cur.execute("drop index if exists idx_tsv")
        cur.execute("select count(*) from pg_indexes where indexname = 'idx_tsv'")
        index_count = cur.fetchone()[0]

        if index_count == 0:
            print("Creating Index (embedding) WITH (num_neighbors=%d, search_list_size=%d, max_alpha=%.2f, use_pq=true, pq_vector_length=%d)"
                   % (self._num_neighbors, self._search_list_size, self._max_alpha, self._pq_vector_length))
            if self._metric == "angular" or self._metric == "euclidean":
                cur.execute(
                            "CREATE INDEX idx_tsv ON items USING tsv (embedding) WITH (num_neighbors = %d, search_list_size = %d, max_alpha=%f, use_pq=true, pq_vector_length=%d)"
                              % (self._num_neighbors, self._search_list_size, self._max_alpha, self._pq_vector_length)
                )
            else:
                raise RuntimeError(f"unknown metric {self._metric}")
            conn.commit()
            print("done!")
        self._cur = cur

    def set_query_arguments(self, query_search_list_size, cte_limit):
        self._query_search_list_size = query_search_list_size
        self._cte_limit = cte_limit
        self._cur.execute("SET tsv.query_search_list_size = %d" % query_search_list_size)
        print("SET tsv.query_search_list_size = %d" % query_search_list_size)
        # TODO set based on available memory
        self._cur.execute("SET work_mem = '256MB'")
        # disable parallel query execution
        self._cur.execute("SET max_parallel_workers_per_gather = 0")
        self._cur.execute("SET enable_seqscan=0")

    def query(self, v, n):
        self._cur.execute(self._query, (v, self._cte_limit, v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def __str__(self):
        return f"TSVectorPQ(num_neighbors={self._num_neighbors}, search_list_size={self._search_list_size}, max_alpha={self._max_alpha}, pq_vector_length={self._pq_vector_length} query_search_list_size={self._query_search_list_size} cte_limit={self._cte_limit})"
