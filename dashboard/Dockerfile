# The hilspy dashboard: Prometheus (receiving OTLP from hilspy) and Perses
# (the dashboard) in one image, for linux/amd64 and linux/arm64. Built and
# pushed to ghcr.io/hakuturu583/hilspy-dashboard:<hilspy version> by
# .github/workflows/dashboard.yml; `tools/setup_host.py --dashboard` pulls it.
#
#   docker run -d --name hilspy-dashboard -p 9090:9090 -p 8080:8080 \
#     -v hilspy-prometheus:/prometheus ghcr.io/hakuturu583/hilspy-dashboard:X.Y.Z
#   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:9090/api/v1/otlp python sim.py
#   open http://localhost:8080/projects/hilspy/dashboards/hilspy
#
# Extra arguments go to Prometheus (--storage.tsdb.retention.time=30d).
# No RUN: every stage is copied, so cross-platform builds need no emulation.

ARG PROMETHEUS_VERSION=v3.15.1
ARG PERSES_VERSION=v0.55.0

FROM persesdev/perses:${PERSES_VERSION} AS perses

# Busybox based: its shell runs the entrypoint.
FROM prom/prometheus:${PROMETHEUS_VERSION}

COPY --from=perses /bin/perses /bin/perses
COPY --from=perses /etc/perses/plugins-archive /etc/perses/plugins-archive
# Written at run time (extracted plugins, the database): owned by the user.
COPY --from=perses --chown=nobody:nobody /etc/perses/plugins /etc/perses/plugins
COPY --from=perses --chown=nobody:nobody /perses /perses
COPY prometheus.yml /etc/prometheus/prometheus.yml
COPY perses/config.yaml /etc/perses/config.yaml
COPY perses/provisioning /etc/perses/provisioning
COPY --chmod=0755 entrypoint.sh /bin/hilspy-dashboard

ARG HILSPY_VERSION=dev
LABEL org.opencontainers.image.title="hilspy-dashboard" \
      org.opencontainers.image.description="Prometheus (OTLP receiver) and Perses with the hilspy dashboard" \
      org.opencontainers.image.source="https://github.com/hakuturu583/hilspy" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${HILSPY_VERSION}"

# The base image's user (nobody), working directory, port 9090 and
# /prometheus volume stay; setting ENTRYPOINT drops its CMD.
EXPOSE 8080
VOLUME ["/perses"]
ENTRYPOINT ["/bin/hilspy-dashboard"]
