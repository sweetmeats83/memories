#!/bin/bash
set -e

# Wait for Postgres to be ready
echo "Waiting for Postgres..."
until (echo > /dev/tcp/${POSTGRES_HOST:-postgres}/${POSTGRES_PORT:-5432}) >/dev/null 2>&1; do
  sleep 1
done
echo "Postgres is up"

# Run migrations
echo "Running Alembic migrations..."
alembic upgrade head
echo "Migrations complete"

# Start the app
exec "$@"
