#!/bin/sh
# Prometheus and Perses side by side; the container ends when either does
# (failing) or on docker stop (cleanly). Arguments go to Prometheus.
signalled=0
trap 'signalled=1' TERM INT

/bin/prometheus \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.enable-otlp-receiver \
    "$@" &
prometheus=$!
/bin/perses --config=/etc/perses/config.yaml --log.level=warning &
perses=$!

# `sleep & wait`, not `sleep`: a trapped signal interrupts `wait` at once.
while [ "$signalled" = 0 ] && kill -0 "$prometheus" 2>/dev/null && kill -0 "$perses" 2>/dev/null; do
    sleep 1 &
    wait $!
done
kill -TERM "$prometheus" "$perses" 2>/dev/null
wait "$prometheus" "$perses"
if [ "$signalled" = 0 ]; then
    echo "hilspy-dashboard: Prometheus or Perses exited; stopping" >&2
    exit 1
fi
