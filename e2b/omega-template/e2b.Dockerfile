FROM e2bdev/base

RUN python -m pip install --no-cache-dir numpy scikit-learn scipy pandas openml
ENV OPENML_CACHE_DIR=/data/openml_cache
RUN mkdir -p /data/openml_cache
RUN mkdir -p /data/benchmarks
RUN mkdir -p /opt/template

COPY data_loader.py /opt/template/data_loader.py
COPY prefetch_openml.py /opt/template/prefetch_openml.py
COPY precompute_benchmarks.py /opt/template/precompute_benchmarks.py

RUN python /opt/template/prefetch_openml.py
RUN python /opt/template/precompute_benchmarks.py
