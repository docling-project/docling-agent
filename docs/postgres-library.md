# PostgreSQL Library Metadata

`docling-agent` can mirror document library metadata into PostgreSQL by setting
`DOCLING_AGENT_LIBRARY_DATABASE_URL`. The converted document payloads are still
stored as `.dclx` files on disk; PostgreSQL stores metadata for lookup and
filtering.

## Storage Layout

By default, converted documents are stored under:

```bash
~/.docling_agent/library/<doc_id>/document_<doc_id>.dclx
```

The local JSON index is stored at:

```bash
~/.docling_agent/library/index.json
```

When PostgreSQL is enabled, metadata is mirrored into:

```text
docling_library_entries
```

PostgreSQL does not read `index.json`. Query and lookup operations use the
PostgreSQL table when `DOCLING_AGENT_LIBRARY_DATABASE_URL` is configured.

Each metadata entry includes document statistics:

```text
document_origin, page_count, table_count, picture_count, text_count, xml_char_count
```

`document_origin` distinguishes converted source documents from generated
documents. Current values are `converted`, `written`, and `in_memory`.
`page_count` is `NULL` for documents without page records. `xml_char_count` is
the number of characters in the DocLang XML stored inside the `.dclx` archive.

## Enable PostgreSQL

Set the database URL before running the CLI:

```bash
export DOCLING_AGENT_LIBRARY_DATABASE_URL="postgresql://user:password@host:5432/dbname"
```

Then run commands normally:

```bash
uv run add-sources ./_model-eval -p ocr-models -c fast
```

## Local PostgreSQL On macOS

### Homebrew

Install and start PostgreSQL:

```bash
brew install postgresql@16
brew services start postgresql@16
```

Create a database:

```bash
createdb docling_agent
```

Use the local database:

```bash
export DOCLING_AGENT_LIBRARY_DATABASE_URL="postgresql:///docling_agent"
```

### Postgres.app

Install Postgres.app from:

```text
https://postgresapp.com
```

Start PostgreSQL from the app, then create a database:

```bash
createdb docling_agent
export DOCLING_AGENT_LIBRARY_DATABASE_URL="postgresql:///docling_agent"
```

## Docker Option

If Docker is available, run:

```bash
docker run --name docling-postgres \
  -e POSTGRES_USER=docling \
  -e POSTGRES_PASSWORD=docling \
  -e POSTGRES_DB=docling_agent \
  -p 5432:5432 \
  -d postgres:16
```

Then configure:

```bash
export DOCLING_AGENT_LIBRARY_DATABASE_URL="postgresql://docling:docling@localhost:5432/docling_agent"
```

## Inspect The Table

For a local database:

```bash
psql "$DOCLING_AGENT_LIBRARY_DATABASE_URL"
```

Then:

```sql
\dt
select doc_id, project_id, name, source_path
from docling_library_entries
limit 10;
```

Stats columns can be used in filters:

```bash
uv run view-sources -f "project_id = 'ocr-models' AND table_count > 0"
uv run view-sources -f "project_id = 'reports' AND document_origin = 'written'"
```

## Clear Library Entries

Clear one project:

```bash
uv run clear-sources --project-id ocr-models --yes
```

Clear the entire library:

```bash
uv run clear-sources --all --yes
```

Clearing removes filesystem document directories and metadata entries. When
PostgreSQL is enabled, matching rows are removed from `docling_library_entries`.
