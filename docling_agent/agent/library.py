"""Document library: persistent storage and status tracking for DoclingDocuments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from docling_core.types.doc.document import DoclingDocument

from docling_agent.logging import logger


class DocStatus(BaseModel):
    """Enrichment status flags for a library document."""

    is_hierarchical: bool = False
    has_summaries: bool = False
    has_keywords: bool = False


class DocLibraryEntry(BaseModel):
    """Metadata record for one document in the library."""

    doc_id: str
    name: str
    source_path: str          # canonical string path of the original file (or "in-memory")
    created_at: str           # ISO-8601 UTC
    updated_at: str           # ISO-8601 UTC
    status: DocStatus = Field(default_factory=DocStatus)
    summary: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class DocLibraryIndex(BaseModel):
    """Top-level index persisted to ``index.json``."""

    entries: dict[str, DocLibraryEntry] = Field(default_factory=dict)   # doc_id → entry
    source_to_id: dict[str, str] = Field(default_factory=dict)          # source_path → doc_id


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _doc_id_for_source(source_path: str) -> str:
    return hashlib.sha256(source_path.encode()).hexdigest()[:16]


def _doc_id_for_name(name: str) -> str:
    """Fallback ID for in-memory / already-loaded documents."""
    return hashlib.sha256(f"mem:{name}:{_now_iso()}".encode()).hexdigest()[:16]


class DoclingLibrary:
    """Manages a directory-based library of converted DoclingDocuments.

    Directory layout::

        <library_path>/
            index.json              ← ``DocLibraryIndex`` (all entries)
            <doc_id>/
                document.json       ← serialized ``DoclingDocument``

    The library is thread-unsafe by design; it is intended for single-process CLI use.
    """

    INDEX_FILE = "index.json"
    DOC_FILE = "document.json"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup_by_source(self, source_path: str) -> Optional[DocLibraryEntry]:
        """Return the entry for *source_path*, or None if not in the library."""
        doc_id = self._index.source_to_id.get(source_path)
        if doc_id:
            return self._index.entries.get(doc_id)
        return None

    def get_entry(self, doc_id: str) -> Optional[DocLibraryEntry]:
        return self._index.entries.get(doc_id)

    def store(
        self,
        doc: DoclingDocument,
        source_path: str,
        *,
        copy_source: bool = False,
    ) -> DocLibraryEntry:
        """Persist *doc* to the library and return its entry.

        If an entry for *source_path* already exists, it is overwritten.
        When *copy_source* is True and *source_path* points to a real file, a
        copy of the file is placed next to ``document.json``.
        """
        doc_id = _doc_id_for_source(source_path)
        doc_dir = self.path / doc_id
        doc_dir.mkdir(exist_ok=True)

        # Write the DoclingDocument JSON
        doc_json = doc.model_dump_json(indent=2)
        (doc_dir / self.DOC_FILE).write_text(doc_json, encoding="utf-8")

        # Optionally copy the original file
        if copy_source:
            src = Path(source_path)
            if src.is_file():
                import shutil
                dest = doc_dir / src.name
                if not dest.exists():
                    shutil.copy2(src, dest)

        # Build / update the index entry
        existing = self._index.entries.get(doc_id)
        entry = DocLibraryEntry(
            doc_id=doc_id,
            name=doc.name,
            source_path=source_path,
            created_at=existing.created_at if existing else _now_iso(),
            updated_at=_now_iso(),
            status=existing.status if existing else DocStatus(),
            summary=existing.summary if existing else None,
            keywords=existing.keywords if existing else [],
            topics=existing.topics if existing else [],
        )
        self._index.entries[doc_id] = entry
        self._index.source_to_id[source_path] = doc_id
        self._save_index()

        logger.debug(f"Library: stored {doc.name!r} → {doc_id} (source={source_path!r})")
        return entry

    def store_in_memory(self, doc: DoclingDocument) -> DocLibraryEntry:
        """Store an in-memory document (no source file) and return its entry."""
        doc_id = _doc_id_for_name(doc.name)
        doc_dir = self.path / doc_id
        doc_dir.mkdir(exist_ok=True)
        (doc_dir / self.DOC_FILE).write_text(doc.model_dump_json(indent=2), encoding="utf-8")

        entry = DocLibraryEntry(
            doc_id=doc_id,
            name=doc.name,
            source_path="in-memory",
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        self._index.entries[doc_id] = entry
        self._save_index()
        return entry

    def load_doc(self, doc_id: str) -> Optional[DoclingDocument]:
        """Load and return the DoclingDocument for *doc_id*, or None."""
        doc_path = self.path / doc_id / self.DOC_FILE
        if not doc_path.exists():
            logger.warning(f"Library: document file missing for {doc_id}")
            return None
        try:
            return DoclingDocument.model_validate_json(doc_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"Library: failed to load {doc_path}: {exc}")
            return None

    def update_status(self, doc_id: str, **flags: bool) -> None:
        """Set status flags on the entry (e.g. ``has_summaries=True``)."""
        entry = self._index.entries.get(doc_id)
        if entry is None:
            logger.warning(f"Library: update_status called for unknown doc_id={doc_id!r}")
            return
        for field, value in flags.items():
            if hasattr(entry.status, field):
                setattr(entry.status, field, value)
        entry.updated_at = _now_iso()
        self._save_index()

    def update_meta(
        self,
        doc_id: str,
        *,
        summary: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        topics: Optional[list[str]] = None,
    ) -> None:
        """Update the document-level summary, keywords, and topics."""
        entry = self._index.entries.get(doc_id)
        if entry is None:
            return
        if summary is not None:
            entry.summary = summary
        if keywords is not None:
            entry.keywords = keywords
        if topics is not None:
            entry.topics = topics
        entry.updated_at = _now_iso()
        self._save_index()

    def resync(self, doc_id: str, doc: DoclingDocument) -> None:
        """Overwrite the stored document JSON (after in-place enrichment)."""
        doc_path = self.path / doc_id / self.DOC_FILE
        if doc_path.exists():
            doc_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
            self.update_status(doc_id)  # just bump updated_at

    def all_entries(self) -> list[DocLibraryEntry]:
        return list(self._index.entries.values())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_index(self) -> DocLibraryIndex:
        index_path = self.path / self.INDEX_FILE
        if index_path.exists():
            try:
                return DocLibraryIndex.model_validate_json(
                    index_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                logger.warning(f"Library: could not load index, starting fresh: {exc}")
        return DocLibraryIndex()

    def _save_index(self) -> None:
        index_path = self.path / self.INDEX_FILE
        index_path.write_text(self._index.model_dump_json(indent=2), encoding="utf-8")
