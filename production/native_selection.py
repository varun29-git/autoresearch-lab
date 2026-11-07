"""Bounded-memory original-row selection for multi-gigabyte corpus indices."""
from __future__ import annotations

import json
import sqlite3


class DiskSelection:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA cache_size=-32768")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA mmap_size=0")
        self.db.execute("CREATE TABLE selected (coordinate TEXT PRIMARY KEY, part INTEGER, uid TEXT, metadata TEXT, seen INTEGER DEFAULT 0) WITHOUT ROWID")
        self.mode = None; self.max_row = None; self.rows = 0

    def add(self, mode, coordinate, part, uid, record):
        if self.mode is not None and self.mode != mode:
            raise ValueError("mixed row-number and UID selection requires a separate source job")
        self.mode = mode
        if mode == "number": self.max_row = max(coordinate, self.max_row or 0)
        # Token counts were already checked against the original bin. Keep
        # only fields actually needed for source and holdout verification.
        metadata = {k: record[k] for k in ("row_key", "n_bytes", "split_family_id", "family_id") if k in record}
        try:
            self.db.execute("INSERT INTO selected(coordinate,part,uid,metadata) VALUES(?,?,?,?)",
                (str(coordinate), part, uid, json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))))
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate selected source row or ambiguous UID across parts") from exc
        self.rows += 1
        if self.rows % 100000 == 0: self.db.commit()

    def ready(self): self.db.commit()

    def find(self, number, aliases):
        coordinates = [str(number)] if self.mode == "number" else sorted(aliases)
        if not coordinates: return None
        query = "SELECT coordinate,part,uid,metadata,seen FROM selected WHERE coordinate IN (" + ",".join("?" for _ in coordinates) + ")"
        matches = self.db.execute(query, coordinates).fetchall()
        if len(matches) > 1: raise ValueError("source row matches conflicting selected identities")
        if not matches: return None
        key, part, uid, metadata, seen = matches[0]
        if seen: raise ValueError("selected UID occurs at multiple raw source coordinates")
        self.db.execute("UPDATE selected SET seen=1 WHERE coordinate=?", (key,))
        return part, uid, json.loads(metadata)

    def close(self): self.db.close()
