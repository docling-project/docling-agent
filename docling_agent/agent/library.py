"""Document library: persistent storage and status tracking for DoclingDocuments."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel, Field

from docling_agent.logging import log_debug, log_error, log_warning


class DocPipelineRun(BaseModel):
    """Pipeline execution recorded for a library document."""

    name: str
    ran_at: str = Field(default_factory=lambda: _now_iso())


class DocEnrichmentRun(BaseModel):
    """Enrichment execution recorded for a library document."""

    name: str
    ran_at: str = Field(default_factory=lambda: _now_iso())
    task: str | None = None


class DocStatus(BaseModel):
    """Processing status for a library document."""

    is_hierarchical: bool = False
    has_summaries: bool = False
    has_keywords: bool = False
    pipelines: list[DocPipelineRun] = Field(default_factory=list)
    enrichments: list[DocEnrichmentRun] = Field(default_factory=list)


class DocStats(BaseModel):
    """Counts and archive-size metadata for a library document."""

    page_count: int | None = None
    table_count: int = 0
    picture_count: int = 0
    text_count: int = 0
    xml_char_count: int = 0


class DocLibraryEntry(BaseModel):
    """Metadata record for one document in the library."""

    doc_id: str
    project_id: str = "default"
    name: str
    source_path: str  # canonical string path of the original file (or "in-memory")
    document_origin: Literal["converted", "written", "in_memory"] = "converted"
    original_mimetype: str | None = None
    doc_path: str
    doc_format: Literal["dclx"] = "dclx"
    created_at: str  # ISO-8601 UTC
    updated_at: str  # ISO-8601 UTC
    status: DocStatus = Field(default_factory=DocStatus)
    stats: DocStats = Field(default_factory=DocStats)
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class DocLibraryIndex(BaseModel):
    """Top-level index persisted to ``index.json``."""

    entries: dict[str, DocLibraryEntry] = Field(default_factory=dict)  # doc_id → entry
    source_to_id: dict[str, str] = Field(default_factory=dict)  # project_id:source_path → doc_id


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _doc_id_for_source(source_path: str, project_id: str) -> str:
    return hashlib.sha256(f"{project_id}:{source_path}".encode()).hexdigest()[:16]


def _doc_id_for_name(name: str, project_id: str) -> str:
    """Fallback ID for in-memory / already-loaded documents."""
    return hashlib.sha256(f"mem:{project_id}:{name}:{_now_iso()}".encode()).hexdigest()[:16]


def _source_key(source_path: str, project_id: str) -> str:
    return f"{project_id}:{source_path}"


def _guess_mimetype(source_path: str) -> str | None:
    if source_path == "in-memory":
        return None
    mimetype, _ = mimetypes.guess_type(source_path)
    return mimetype


class DoclingLibrary:
    """Manages a document library of converted DoclingDocuments.

    Directory layout::

        <library_path>/
            index.json              ← ``DocLibraryIndex`` (all entries)
            <doc_id>/
                document_<doc_id>.dclx ← serialized ``DoclingDocument``

    When ``database_url`` or ``DOCLING_AGENT_LIBRARY_DATABASE_URL`` is set,
    entry metadata is mirrored to PostgreSQL and query methods run in SQL.
    The document payloads remain filesystem-backed as dclx archives.
    """

    INDEX_FILE = "index.json"
    LEGACY_DOC_FILE = "document.dclx"
    PG_TABLE = "docling_library_entries"

    def __init__(self, path: Path, *, project_id: str = "default", database_url: str | None = None) -> None:
        self.path = path
        self.project_id = project_id or "default"
        self.database_url = database_url or os.environ.get("DOCLING_AGENT_LIBRARY_DATABASE_URL")
        self.path.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()
        if self.database_url:
            self._ensure_pg_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup_by_source(self, source_path: str, *, project_id: str | None = None) -> DocLibraryEntry | None:
        """Return the entry for *source_path*, or None if not in the library."""
        project_id = project_id or self.project_id
        if self.database_url:
            return self._pg_lookup_by_source(source_path=source_path, project_id=project_id)
        doc_id = self._index.source_to_id.get(_source_key(source_path, project_id))
        if doc_id:
            return self._index.entries.get(doc_id)
        return None

    def get_entry(self, doc_id: str) -> DocLibraryEntry | None:
        if self.database_url:
            return self._pg_get_entry(doc_id)
        return self._index.entries.get(doc_id)

    def store(
        self,
        doc: DoclingDocument,
        source_path: str,
        *,
        copy_source: bool = False,
        project_id: str | None = None,
        original_mimetype: str | None = None,
        conversion_pipeline: str | None = None,
        document_origin: Literal["converted", "written", "in_memory"] = "converted",
    ) -> DocLibraryEntry:
        """Persist *doc* to the library and return its entry.

        If an entry for *source_path* already exists, it is overwritten.
        When *copy_source* is True and *source_path* points to a real file, a
        copy of the file is placed next to the ``document_<doc_id>.dclx`` archive.
        """
        project_id = project_id or self.project_id
        doc_id = _doc_id_for_source(source_path, project_id)
        doc_dir = self.path / doc_id
        doc_dir.mkdir(exist_ok=True)

        doc_path = self._doc_path_for_id(doc_id)

        # Write the DoclingDocument as DocLang archive (dclx).
        self._save_doc(doc=doc, doc_path=doc_path)
        stats = self._stats_for_doc(doc=doc, doc_path=doc_path)

        # Optionally copy the original file
        if copy_source:
            src = Path(source_path)
            if src.is_file():
                import shutil

                dest = doc_dir / src.name
                if not dest.exists():
                    shutil.copy2(src, dest)

        # Build / update the index entry
        existing = self.get_entry(doc_id)
        status = existing.status if existing else DocStatus()
        if conversion_pipeline:
            self._record_pipeline(status, conversion_pipeline)

        entry = DocLibraryEntry(
            doc_id=doc_id,
            project_id=project_id,
            name=doc.name,
            source_path=source_path,
            document_origin=document_origin,
            original_mimetype=original_mimetype
            or (existing.original_mimetype if existing else _guess_mimetype(source_path)),
            doc_path=str(doc_path),
            created_at=existing.created_at if existing else _now_iso(),
            updated_at=_now_iso(),
            status=status,
            stats=stats,
            summary=existing.summary if existing else None,
            keywords=existing.keywords if existing else [],
            topics=existing.topics if existing else [],
        )
        self._persist_entry(entry)

        log_debug(f"Library: stored {doc.name!r} → {doc_id} (source={source_path!r})")
        return entry

    def store_in_memory(
        self,
        doc: DoclingDocument,
        *,
        project_id: str | None = None,
        document_origin: Literal["written", "in_memory"] = "in_memory",
    ) -> DocLibraryEntry:
        """Store an in-memory document (no source file) and return its entry."""
        project_id = project_id or self.project_id
        doc_id = _doc_id_for_name(doc.name, project_id)
        doc_dir = self.path / doc_id
        doc_dir.mkdir(exist_ok=True)
        doc_path = self._doc_path_for_id(doc_id)
        self._save_doc(doc=doc, doc_path=doc_path)
        stats = self._stats_for_doc(doc=doc, doc_path=doc_path)

        entry = DocLibraryEntry(
            doc_id=doc_id,
            project_id=project_id,
            name=doc.name,
            source_path=f"{document_origin}:{doc_id}",
            document_origin=document_origin,
            original_mimetype=None,
            doc_path=str(doc_path),
            created_at=_now_iso(),
            updated_at=_now_iso(),
            stats=stats,
        )
        self._persist_entry(entry)
        return entry

    def load_doc(self, doc_id: str) -> DoclingDocument | None:
        """Load and return the DoclingDocument for *doc_id*, or None."""
        entry = self.get_entry(doc_id)
        doc_path = Path(entry.doc_path) if entry else self._doc_path_for_id(doc_id)
        if not doc_path.exists():
            log_warning(f"Library: document file missing for {doc_id}")
            return None
        try:
            doc = DoclingDocument.load_from_doclang_archive(doc_path)
            if entry is not None:
                doc.name = entry.name
            return doc
        except Exception as exc:
            log_error(f"Library: failed to load {doc_path}: {exc}")
            return None

    def update_status(self, doc_id: str, **flags: bool) -> None:
        """Set status flags on the entry (e.g. ``has_summaries=True``)."""
        entry = self.get_entry(doc_id)
        if entry is None:
            log_warning(f"Library: update_status called for unknown doc_id={doc_id!r}")
            return
        for field, value in flags.items():
            if hasattr(entry.status, field):
                setattr(entry.status, field, value)
        entry.updated_at = _now_iso()
        self._persist_entry(entry)

    def record_enrichments(self, doc_id: str, enrichments: list[str], *, task: str | None = None) -> None:
        """Record enrichment operations that have been applied to the document."""
        entry = self.get_entry(doc_id)
        if entry is None:
            log_warning(f"Library: record_enrichments called for unknown doc_id={doc_id!r}")
            return
        for enrichment in enrichments:
            self._record_enrichment(entry.status, enrichment, task=task)
        entry.updated_at = _now_iso()
        self._persist_entry(entry)

    def update_meta(
        self,
        doc_id: str,
        *,
        summary: str | None = None,
        keywords: list[str] | None = None,
        topics: list[str] | None = None,
    ) -> None:
        """Update the document-level summary, keywords, and topics."""
        entry = self.get_entry(doc_id)
        if entry is None:
            return
        if summary is not None:
            entry.summary = summary
        if keywords is not None:
            entry.keywords = keywords
        if topics is not None:
            entry.topics = topics
        entry.updated_at = _now_iso()
        self._persist_entry(entry)

    def resync(self, doc_id: str, doc: DoclingDocument) -> None:
        """Overwrite the stored dclx archive (after in-place enrichment)."""
        entry = self.get_entry(doc_id)
        doc_path = Path(entry.doc_path) if entry else self._doc_path_for_id(doc_id)
        if doc_path.exists():
            self._save_doc(doc=doc, doc_path=doc_path)
            if entry is not None:
                entry.stats = self._stats_for_doc(doc=doc, doc_path=doc_path)
                entry.updated_at = _now_iso()
                self._persist_entry(entry)
            self.update_status(doc_id)  # just bump updated_at

    def all_entries(self) -> list[DocLibraryEntry]:
        if self.database_url:
            return self._pg_all_entries()
        return list(self._index.entries.values())

    def query_entries(
        self,
        *,
        project_id: str | None = None,
        name: str | None = None,
        source_path: str | None = None,
        mimetype: str | None = None,
        document_origin: Literal["converted", "written", "in_memory"] | None = None,
        pipeline: str | None = None,
        enrichments: list[str] | None = None,
        text: str | None = None,
        limit: int = 50,
    ) -> list[DocLibraryEntry]:
        """Query library entries, using PostgreSQL when configured."""
        project_id = project_id or self.project_id
        if self.database_url:
            return self._pg_query_entries(
                project_id=project_id,
                name=name,
                source_path=source_path,
                mimetype=mimetype,
                document_origin=document_origin,
                pipeline=pipeline,
                enrichments=enrichments,
                text=text,
                limit=limit,
            )

        entries = [entry for entry in self._index.entries.values() if entry.project_id == project_id]
        if name is not None:
            entries = [entry for entry in entries if name.lower() in entry.name.lower()]
        if source_path is not None:
            entries = [entry for entry in entries if entry.source_path == source_path]
        if mimetype is not None:
            entries = [entry for entry in entries if entry.original_mimetype == mimetype]
        if document_origin is not None:
            entries = [entry for entry in entries if entry.document_origin == document_origin]
        if pipeline is not None:
            entries = [entry for entry in entries if any(run.name == pipeline for run in entry.status.pipelines)]
        if enrichments:
            requested = set(enrichments)
            entries = [entry for entry in entries if requested.issubset({run.name for run in entry.status.enrichments})]
        if text is not None:
            needle = text.lower()
            entries = [
                entry
                for entry in entries
                if needle in entry.name.lower()
                or (entry.summary is not None and needle in entry.summary.lower())
                or any(needle in keyword.lower() for keyword in entry.keywords)
            ]
        entries.sort(key=lambda entry: entry.updated_at, reverse=True)
        return entries[:limit]

    def query_entries_by_postgres_filter(self, postgres_filter: str, *, limit: int = 100) -> list[DocLibraryEntry]:
        """Query entries with a raw PostgreSQL WHERE predicate."""
        if not self.database_url:
            raise RuntimeError("PostgreSQL filtering requires DOCLING_AGENT_LIBRARY_DATABASE_URL.")
        self._validate_postgres_filter(postgres_filter)
        return self._pg_query_entries_by_filter(postgres_filter=postgres_filter, limit=limit)

    def clear(self, *, project_id: str | None = None, all_projects: bool = False) -> int:
        """Remove entries and stored archives for a project or the entire library.

        Returns the number of entries removed.
        """
        if not all_projects:
            project_id = project_id or self.project_id

        entries = self.all_entries() if all_projects else self.query_entries(project_id=project_id, limit=10**9)
        removed = 0
        for entry in entries:
            doc_dir = Path(entry.doc_path).parent
            if doc_dir.is_dir():
                shutil.rmtree(doc_dir)
            self._index.entries.pop(entry.doc_id, None)
            self._index.source_to_id.pop(_source_key(entry.source_path, entry.project_id), None)
            removed += 1

        if all_projects:
            self._index.entries.clear()
            self._index.source_to_id.clear()

        self._save_index()
        if self.database_url:
            self._pg_clear(project_id=project_id, all_projects=all_projects)
        return removed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _doc_path_for_id(self, doc_id: str) -> Path:
        return self.path / doc_id / f"document_{doc_id}.dclx"

    def _load_index(self) -> DocLibraryIndex:
        index_path = self.path / self.INDEX_FILE
        if index_path.exists():
            try:
                return DocLibraryIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
            except Exception as exc:
                log_warning(f"Library: could not load index, starting fresh: {exc}")
        return DocLibraryIndex()

    def _save_index(self) -> None:
        index_path = self.path / self.INDEX_FILE
        index_path.write_text(self._index.model_dump_json(indent=2), encoding="utf-8")

    def _persist_entry(self, entry: DocLibraryEntry) -> None:
        self._index.entries[entry.doc_id] = entry
        self._index.source_to_id[_source_key(entry.source_path, entry.project_id)] = entry.doc_id
        self._save_index()
        if self.database_url:
            self._pg_upsert_entry(entry)

    def _save_doc(self, *, doc: DoclingDocument, doc_path: Path) -> None:
        if not hasattr(doc, "save_as_doclang_archive"):
            raise RuntimeError("The installed docling-core version does not support dclx archives.")
        doc.save_as_doclang_archive(doc_path)

    def _stats_for_doc(self, *, doc: DoclingDocument, doc_path: Path) -> DocStats:
        page_count = doc.num_pages()
        return DocStats(
            page_count=page_count or None,
            table_count=len(doc.tables),
            picture_count=len(doc.pictures),
            text_count=len(doc.texts),
            xml_char_count=self._xml_char_count_for_archive(doc_path),
        )

    def _xml_char_count_for_archive(self, doc_path: Path) -> int:
        try:
            with zipfile.ZipFile(doc_path) as archive:
                xml_name = "document.xml"
                if xml_name not in archive.namelist():
                    xml_name = next((name for name in archive.namelist() if name.endswith(".xml")), "")
                if not xml_name:
                    return 0
                return len(archive.read(xml_name).decode("utf-8"))
        except Exception as exc:
            log_warning(f"Library: could not count DocLang XML characters for {doc_path}: {exc}")
            return 0

    def _record_pipeline(self, status: DocStatus, pipeline: str) -> None:
        if any(run.name == pipeline for run in status.pipelines):
            return
        status.pipelines.append(DocPipelineRun(name=pipeline))

    def _record_enrichment(self, status: DocStatus, enrichment: str, *, task: str | None = None) -> None:
        if any(run.name == enrichment for run in status.enrichments):
            return
        status.enrichments.append(DocEnrichmentRun(name=enrichment, task=task))

    def _validate_postgres_filter(self, postgres_filter: str) -> None:
        if not postgres_filter.strip():
            raise ValueError("PostgreSQL filter must not be empty.")
        forbidden = (";", "--", "/*", "*/")
        if any(token in postgres_filter for token in forbidden):
            raise ValueError("PostgreSQL filter must be a single WHERE predicate.")

    def _connect_pg(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("PostgreSQL library storage requires the 'psycopg[binary]' dependency.") from exc

        return psycopg.connect(self.database_url)

    def _ensure_pg_table(self) -> None:
        with self._connect_pg() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.PG_TABLE} (
                    doc_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL DEFAULT 'default',
                    source_path TEXT NOT NULL,
                    document_origin TEXT NOT NULL DEFAULT 'converted',
                    name TEXT NOT NULL,
                    original_mimetype TEXT,
                    doc_path TEXT NOT NULL,
                    page_count INTEGER,
                    table_count INTEGER NOT NULL DEFAULT 0,
                    picture_count INTEGER NOT NULL DEFAULT 0,
                    text_count INTEGER NOT NULL DEFAULT 0,
                    xml_char_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    entry JSONB NOT NULL,
                    UNIQUE (project_id, source_path)
                )
                """
            )
            conn.execute(f"ALTER TABLE {self.PG_TABLE} ADD COLUMN IF NOT EXISTS page_count INTEGER")
            conn.execute(
                f"ALTER TABLE {self.PG_TABLE} ADD COLUMN IF NOT EXISTS document_origin TEXT NOT NULL DEFAULT 'converted'"
            )
            conn.execute(f"ALTER TABLE {self.PG_TABLE} ADD COLUMN IF NOT EXISTS table_count INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                f"ALTER TABLE {self.PG_TABLE} ADD COLUMN IF NOT EXISTS picture_count INTEGER NOT NULL DEFAULT 0"
            )
            conn.execute(f"ALTER TABLE {self.PG_TABLE} ADD COLUMN IF NOT EXISTS text_count INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                f"ALTER TABLE {self.PG_TABLE} ADD COLUMN IF NOT EXISTS xml_char_count INTEGER NOT NULL DEFAULT 0"
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS {self.PG_TABLE}_project_idx ON {self.PG_TABLE} (project_id)")
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.PG_TABLE}_source_idx ON {self.PG_TABLE} (project_id, source_path)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.PG_TABLE}_mimetype_idx ON {self.PG_TABLE} (original_mimetype)"
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS {self.PG_TABLE}_origin_idx ON {self.PG_TABLE} (document_origin)")
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.PG_TABLE}_entry_gin_idx ON {self.PG_TABLE} USING GIN (entry)"
            )

    def _pg_upsert_entry(self, entry: DocLibraryEntry) -> None:
        from psycopg.types.json import Jsonb

        with self._connect_pg() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.PG_TABLE} (
                    doc_id, project_id, source_path, document_origin, name, original_mimetype,
                    doc_path, page_count, table_count, picture_count, text_count,
                    xml_char_count, created_at, updated_at, entry
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::timestamptz, %s::timestamptz, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    project_id = EXCLUDED.project_id,
                    source_path = EXCLUDED.source_path,
                    document_origin = EXCLUDED.document_origin,
                    name = EXCLUDED.name,
                    original_mimetype = EXCLUDED.original_mimetype,
                    doc_path = EXCLUDED.doc_path,
                    page_count = EXCLUDED.page_count,
                    table_count = EXCLUDED.table_count,
                    picture_count = EXCLUDED.picture_count,
                    text_count = EXCLUDED.text_count,
                    xml_char_count = EXCLUDED.xml_char_count,
                    updated_at = EXCLUDED.updated_at,
                    entry = EXCLUDED.entry
                """,
                (
                    entry.doc_id,
                    entry.project_id,
                    entry.source_path,
                    entry.document_origin,
                    entry.name,
                    entry.original_mimetype,
                    entry.doc_path,
                    entry.stats.page_count,
                    entry.stats.table_count,
                    entry.stats.picture_count,
                    entry.stats.text_count,
                    entry.stats.xml_char_count,
                    entry.created_at,
                    entry.updated_at,
                    Jsonb(entry.model_dump(mode="json")),
                ),
            )

    def _pg_entry_from_row(self, row: tuple[Any, ...] | None) -> DocLibraryEntry | None:
        if row is None:
            return None
        return DocLibraryEntry.model_validate(row[0])

    def _pg_lookup_by_source(self, *, source_path: str, project_id: str) -> DocLibraryEntry | None:
        with self._connect_pg() as conn:
            row = conn.execute(
                f"SELECT entry FROM {self.PG_TABLE} WHERE project_id = %s AND source_path = %s",
                (project_id, source_path),
            ).fetchone()
        return self._pg_entry_from_row(row)

    def _pg_get_entry(self, doc_id: str) -> DocLibraryEntry | None:
        with self._connect_pg() as conn:
            row = conn.execute(f"SELECT entry FROM {self.PG_TABLE} WHERE doc_id = %s", (doc_id,)).fetchone()
        return self._pg_entry_from_row(row)

    def _pg_all_entries(self) -> list[DocLibraryEntry]:
        with self._connect_pg() as conn:
            rows = conn.execute(f"SELECT entry FROM {self.PG_TABLE} ORDER BY updated_at DESC").fetchall()
        return [DocLibraryEntry.model_validate(row[0]) for row in rows]

    def _pg_query_entries(
        self,
        *,
        project_id: str,
        name: str | None,
        source_path: str | None,
        mimetype: str | None,
        document_origin: Literal["converted", "written", "in_memory"] | None,
        pipeline: str | None,
        enrichments: list[str] | None,
        text: str | None,
        limit: int,
    ) -> list[DocLibraryEntry]:
        from psycopg.types.json import Jsonb

        where = ["project_id = %s"]
        params: list[Any] = [project_id]

        if name is not None:
            where.append("name ILIKE %s")
            params.append(f"%{name}%")
        if source_path is not None:
            where.append("source_path = %s")
            params.append(source_path)
        if mimetype is not None:
            where.append("original_mimetype = %s")
            params.append(mimetype)
        if document_origin is not None:
            where.append("document_origin = %s")
            params.append(document_origin)
        if pipeline is not None:
            where.append("entry @> %s::jsonb")
            params.append(Jsonb({"status": {"pipelines": [{"name": pipeline}]}}))
        if enrichments:
            for enrichment in enrichments:
                where.append("entry @> %s::jsonb")
                params.append(Jsonb({"status": {"enrichments": [{"name": enrichment}]}}))
        if text is not None:
            where.append(
                "(name ILIKE %s OR entry->>'summary' ILIKE %s OR EXISTS "
                "(SELECT 1 FROM jsonb_array_elements_text(entry->'keywords') kw WHERE kw ILIKE %s))"
            )
            params.extend([f"%{text}%", f"%{text}%", f"%{text}%"])

        params.append(limit)
        with self._connect_pg() as conn:
            rows = conn.execute(
                f"""
                SELECT entry
                FROM {self.PG_TABLE}
                WHERE {" AND ".join(where)}
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                params,
            ).fetchall()
        return [DocLibraryEntry.model_validate(row[0]) for row in rows]

    def _pg_query_entries_by_filter(self, *, postgres_filter: str, limit: int) -> list[DocLibraryEntry]:
        with self._connect_pg() as conn:
            rows = conn.execute(
                f"""
                SELECT entry
                FROM {self.PG_TABLE}
                WHERE {postgres_filter}
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [DocLibraryEntry.model_validate(row[0]) for row in rows]

    def _pg_clear(self, *, project_id: str | None, all_projects: bool) -> None:
        with self._connect_pg() as conn:
            if all_projects:
                conn.execute(f"DELETE FROM {self.PG_TABLE}")
            else:
                conn.execute(f"DELETE FROM {self.PG_TABLE} WHERE project_id = %s", (project_id,))
