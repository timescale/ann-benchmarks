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

USER postgres
RUN service postgresql start && \
    psql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann'" && \
    psql -c "CREATE DATABASE ann" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    psql -d ann -c "GRANT ALL ON SCHEMA public TO ann" && \
    psql -d ann -c "CREATE EXTENSION vector" && \
    psql -c "ALTER USER ann SET maintenance_work_mem = '250GB'" && \
    psql -c "ALTER USER ann SET max_parallel_maintenance_workers = 32" && \
    psql -c "ALTER SYSTEM SET max_worker_processes = 40" && \
    psql -c "ALTER SYSTEM SET max_parallel_workers = 32" && \
    psql -c "ALTER SYSTEM SET shared_buffers = '64GB'" && \
    psql -c "ALTER SYSTEM SET effective_cache_size = '400GB'" && \
    psql -c "ALTER SYSTEM SET random_page_cost = 1.1" && \
    psql -c "ALTER SYSTEM SET max_parallel_workers_per_gather = 0"
USER root

RUN pip install psycopg[binary] pgvector
