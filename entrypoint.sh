#!/usr/bin/env bash
set -e

# 1. Run the vector database generation script first as requested
echo "[ENTRYPOINT] Starting vector database generation..."
python scripts/generate_vector_db.py
echo "[ENTRYPOINT] Vector database generation completed successfully."

# 2. If arguments/commands are passed, execute them; otherwise keep container alive in background
if [ "$#" -gt 0 ]; then
    echo "[ENTRYPOINT] Executing custom command: $@"
    exec "$@"
else
    echo "[ENTRYPOINT] Container is ready and running in background."
    exec tail -f /dev/null
fi
