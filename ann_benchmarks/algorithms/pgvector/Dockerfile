FROM ann-benchmarks

# https://github.com/pgvector/pgvector/blob/master/Dockerfile

RUN git clone --branch v0.8.5 --depth 1 https://github.com/pgvector/pgvector /tmp/pgvector

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends build-essential postgresql-common
RUN /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
RUN apt-get install -y --no-install-recommends postgresql-18 postgresql-server-dev-18
RUN sh -c 'echo "local all all trust" > /etc/postgresql/18/main/pg_hba.conf'

# Dynamically set OPTFLAGS based on the architecture
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        OPTFLAGS="-march=native -msve-vector-bits=512"; \
    elif [ "$ARCH" = "x86_64" ]; then \
        OPTFLAGS="-march=native -mprefer-vector-width=512"; \
    else \
        OPTFLAGS="-march=native"; \
    fi && \
    cd /tmp/pgvector && \
    make clean && \
    make OPTFLAGS="$OPTFLAGS" && \
    make install

# Deterministic port + no TCP: builds and runs use host
# networking, so cluster creation otherwise inherits whatever
# port is free next to the HOST's PostgreSQL, and a TCP bind
# would collide with it. The benchmark connects over the Unix
# socket.
RUN pg_conftool 18 main set port 5432 && \
    pg_conftool 18 main set listen_addresses ''

# Benchmark-scale knobs. Defaults match the upstream out-of-the-box
# image (modest memory, serial index builds). Large-dataset runs (10M+)
# opt in at build time, e.g.:
#   docker build --build-arg MAINTENANCE_WORK_MEM=250GB \
#     --build-arg SHARED_BUFFERS=64GB --build-arg EFFECTIVE_CACHE_SIZE=400GB \
#     --build-arg MAX_PARALLEL_MAINTENANCE_WORKERS=32 ...
ARG MAINTENANCE_WORK_MEM=4GB
ARG SHARED_BUFFERS=4GB
ARG EFFECTIVE_CACHE_SIZE=16GB
ARG MAX_PARALLEL_MAINTENANCE_WORKERS=0

USER postgres
RUN service postgresql start && \
    psql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann'" && \
    psql -c "CREATE DATABASE ann" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    psql -d ann -c "GRANT ALL ON SCHEMA public TO ann" && \
    psql -d ann -c "CREATE EXTENSION vector" && \
    psql -c "ALTER USER ann SET maintenance_work_mem = '${MAINTENANCE_WORK_MEM}'" && \
    psql -c "ALTER USER ann SET max_parallel_maintenance_workers = ${MAX_PARALLEL_MAINTENANCE_WORKERS}" && \
    psql -c "ALTER SYSTEM SET max_worker_processes = 40" && \
    psql -c "ALTER SYSTEM SET max_parallel_workers = 32" && \
    psql -c "ALTER SYSTEM SET shared_buffers = '${SHARED_BUFFERS}'" && \
    psql -c "ALTER SYSTEM SET effective_cache_size = '${EFFECTIVE_CACHE_SIZE}'" && \
    psql -c "ALTER SYSTEM SET random_page_cost = 1.1" && \
    psql -c "ALTER SYSTEM SET max_parallel_workers_per_gather = 0"
USER root

RUN pip install psycopg[binary] pgvector
