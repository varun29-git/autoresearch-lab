"""Immutable view of a SHA-verified compact ownership checkpoint.

Fixed materializers never allocate records. They can retain the predecessor
artifact directly instead of copying it into RAM and publishing another copy.
The caller verifies the complete file digest before opening this view.
"""
from pathlib import Path
import sqlite3

from production.compact_ledger import CompactLedger


class ReadOnlyLedger(CompactLedger):
    def __init__(self, predecessor, *, cache_mib=512, mmap_mib=2048):
        self.source=Path(predecessor).resolve()
        self.memory=False;self.seen=None;self.readonly=True
        if cache_mib<=0 or mmap_mib<0:raise ValueError('invalid immutable ledger cache')
        self.conn=sqlite3.connect(self.source.as_uri()+'?mode=ro&immutable=1',uri=True)
        try:
            self.conn.execute('PRAGMA query_only=ON')
            self.conn.execute('PRAGMA cache_size=-'+str(cache_mib*1024))
            self.conn.execute('PRAGMA mmap_size='+str(mmap_mib<<20))
            if self.conn.execute('PRAGMA quick_check').fetchone()[0]!='ok':
                raise ValueError('invalid immutable ownership ledger')
            meta=dict(self.conn.execute('SELECT name,value FROM metadata'))
            if meta.get('schema')!='ci1_compact_ledger_v2':
                raise ValueError('immutable ledger requires the compact schema')
            self.count=int(meta['count'])
            if self.count<0:raise ValueError('invalid immutable ledger record count')
            self.names={value:n for n,value in self.conn.execute('SELECT id,value FROM names')}
            self.values={n:value for value,n in self.names.items()}
        except BaseException:
            self.conn.close();raise

    def _forbid_write(self,*args,**kwargs):
        raise ValueError('fixed materialization cannot mutate its immutable ownership ledger')

    add_many=change_owner=mark_drained=commit=snapshot=name=configure=_forbid_write
