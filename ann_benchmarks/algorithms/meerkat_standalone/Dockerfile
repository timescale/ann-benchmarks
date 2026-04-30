FROM ann-benchmarks

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libblis-dev clang

# Install meson + ninja for building
RUN pip install meson ninja

# Build meerkat from local source (tarball created by install.py via git archive)
ADD ann_benchmarks/algorithms/meerkat_standalone/src.tar.gz /tmp/meerkat

# Build with clang to avoid GCC ifunc resolver crashes with
# target_clones at dlopen time. Clang handles target_clones
# via inline dispatch which is safe for shared libraries.
RUN CC=clang CFLAGS="-Wno-unused-function" \
    meson setup /tmp/meerkat/builddir /tmp/meerkat \
        --buildtype=release \
        -Dnative=true \
        -Dpostgresql=disabled && \
    ninja -C /tmp/meerkat/builddir libmkt.so && \
    cp /tmp/meerkat/builddir/libmkt.so /usr/local/lib/ && \
    ldconfig
