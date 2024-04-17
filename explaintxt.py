#!/usr/bin/env python3

# get an explain plan
import h5py
import psycopg
from pgvector.psycopg import register_vector
import random
import re
import numpy as np

DATASET = "data/cohere-wikipedia-22-12-50M-angular.hdf5"
QUERY = "select id from public.items order by embedding <=> %s limit %s"
LIMIT = 10
SETTINGS = [
    "set local tsv.query_search_list_size = 25;",
    "set local tsv.query_resort = 25;",
    "set local work_mem = '512MB';",
    "set local maintenance_work_mem = '8GB';",
    "set local max_parallel_workers_per_gather = 0;",
    "set local enable_seqscan = 0;",
    "set jit = 'off';",
    "set local log_min_messages='DEBUG1';",
]


f = h5py.File(DATASET, 'r')
t = f['test']
v = t[random.randrange(start=0, stop=t.shape[0])]
v = np.array([float(n) for n in v])
q = "explain (analyze, verbose, costs, buffers) " + QUERY
with psycopg.connect("postgres://ann:ann@localhost:5432/ann") as con:
    register_vector(con)
    with con.cursor() as cur:
        with con.transaction():
            for setting in SETTINGS:
                cur.execute(setting)
            cur.execute(q, (v, LIMIT), binary=True, prepare=True)
            explain = "\n".join([r[0] for r in cur.fetchall()])
explain = re.sub("'\[([0-9\.\-,])*\]'::vector", "[...]::vector", explain)

with open('explain.txt', 'w') as w:
    for setting in SETTINGS:
        w.write(setting + "\n")
    w.write(q + "\n")
    w.write(explain)
    w.flush()
    w.close()

