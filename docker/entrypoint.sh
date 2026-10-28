#!/bin/sh
# Start the HUD. With F1PRED_DEMO=1 (the default) it serves the synthetic demo,
# building it first if needed. With F1PRED_DEMO=0 it serves the real warehouse
# in $F1PRED_DATA_DIR, which needs `f1pred ingest/load/transform/evaluate` first.
# Extra arguments go to `f1pred serve`, e.g. --openf1-session latest --total-laps 57.
set -eu

if [ "${F1PRED_DEMO:-1}" = "1" ]; then
    if [ ! -f "$F1PRED_DATA_DIR/demo/reports/model/models/lap.joblib" ]; then
        echo "Building the demo warehouse and models (first start only, a few minutes)..."
        f1pred demo --seasons 2019-2026
        f1pred --demo transform
        f1pred --demo evaluate --quick
    fi
    exec f1pred --demo serve --host 0.0.0.0 --port "${PORT:-8000}" "$@"
fi
exec f1pred serve --host 0.0.0.0 --port "${PORT:-8000}" "$@"
