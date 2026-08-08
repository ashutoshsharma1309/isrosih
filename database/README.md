# database/

PostgreSQL 16 + PostGIS.

- [`schema.sql`](schema.sql) — initial schema: regions (with geometries),
  weather observations, satellite image catalog, persisted predictions
  (with JSONB explanation payloads), and alerts. Auto-applied on first
  container start via docker-compose.
- `migrations/` — Alembic migrations for all schema changes after v1
  (configured in Phase 7 when the backend gets real DB access).

Start a local database with:

```bash
docker compose -f deployment/docker-compose.yml up db
```
