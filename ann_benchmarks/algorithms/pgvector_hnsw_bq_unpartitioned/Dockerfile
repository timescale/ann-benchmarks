FROM ann-benchmarks

RUN DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends
USER root
RUN python3 -m pip install "psycopg[binary, pool]" pgvector
