#!/usr/bin/env python3

import h5py
import psycopg
from pgvector.psycopg import register_vector
import numpy as np

DATASET = "data/cohere-wikipedia-22-12-50M-angular.hdf5"

f = h5py.File(DATASET, 'r')
t = f['test']
with psycopg.connect("postgres://ann:ann@localhost:5432/ann") as con:
    register_vector(con)
    with con.cursor(binary=True) as cur:
        with cur.copy("copy public.test (embedding) from stdin (format binary)") as cpy:
            cpy.set_types(['vector'])
            for v in t:
                cpy.write_row((np.array([float(n) for n in v]), ))
    con.commit()

print("done")
