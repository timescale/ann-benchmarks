#!/usr/bin/env python3

# get an explain plan
import h5py
import psycopg
import random
import json
import re
query_list_search_size = 25
limit = 10
f = h5py.File('data/cohere-wikipedia-22-12-50M-angular.hdf5', 'r')
t = f['test']
v = t[random.randrange(start=0, stop=t.shape[0])]
v = [float(n) for n in v]
q = f"select id from public.items order by embedding <=> %s::vector limit %s"
#q = f"with x as materialized (select id, embedding <=> %s::vector as distance from public.items order by 2 limit 100) select id from x order by distance limit %s"
set_qsls = f"set local tsv.query_search_list_size = {query_list_search_size}"
q = "explain (analyze, verbose, costs, buffers) " + q
#explain: str = ""
with psycopg.connect("postgres://ann:ann@localhost:5432/ann") as con:
    with con.cursor() as cur:
        with con.transaction():
            cur.execute(set_qsls)
            cur.execute("set local tsv.query_resort = 25")
            cur.execute("set local work_mem = '8GB'")
            cur.execute("set local max_parallel_workers_per_gather = 4")
            cur.execute("set local enable_seqscan=0")
            cur.execute("set jit = 'off'")
            cur.execute("set local log_min_messages='DEBUG1'")
            cur.execute(q, (v, limit), binary=True, prepare=True)
            explain = "\n".join([r[0] for r in cur.fetchall()])
explain = re.sub("'\[([0-9\.\-,])*\]'::vector", "[...]::vector", explain)
with open('explain.txt', 'w') as w:
    w.write(set_qsls + "\n")
    w.write(q + "\n")
    w.write(explain)
    w.flush()
    w.close()

