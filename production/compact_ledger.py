"""Disposable, compact allocation ledger with a verified checkpoint boundary.

Only immutable S3 inputs and a completed snapshot are durable state. Working
pages stay in RAM where they fit; large ledgers use a bounded scratch fallback.
Original row identities remain in source/segment manifests and are bound here
by SHA-256, rather than repeating long source paths in every database record.
"""
from __future__ import annotations

import hashlib
from itertools import chain
from pathlib import Path
import sqlite3
import sys


def key(value):
    if len(value) != 64:
        raise ValueError('invalid source SHA-256')
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError('invalid source SHA-256') from exc
    if len(result) != 32:
        raise ValueError('invalid source SHA-256')
    return result


def row_key_hash(value):
    return hashlib.sha256(value.encode('utf-8')).digest()


class CompactLedger:
    def __init__(self, work, predecessor=None, *, memory_mib=73728,
                 seen_mib=12288, cache_mib=24576, use_seen=True,
                 commit_rows=0, bulk_insert_rows=0):
        self.work=Path(work);self.work.mkdir(parents=True,exist_ok=True)
        self.memory_limit=memory_mib<<20;self.seen_limit=seen_mib<<20
        self.cache_mib=cache_mib;self.memory=True;self.seen=None
        if commit_rows<0 or not 0<=bulk_insert_rows<=200:raise ValueError('invalid ledger batching')
        self.commit_rows=commit_rows;self.bulk_insert_rows=bulk_insert_rows;self.uncommitted_rows=0
        source=None
        if predecessor:
            source=sqlite3.connect('file:'+str(Path(predecessor).resolve())+'?mode=ro',uri=True)
            if source.execute('PRAGMA quick_check').fetchone()[0]!='ok':
                raise ValueError('invalid predecessor ledger')
            compact=source.execute("SELECT 1 FROM sqlite_master WHERE name='allocations' AND type='table'").fetchone()
        else:
            compact=False
        if compact and Path(predecessor).stat().st_size>self.memory_limit:
            self.conn=sqlite3.connect(self.work/'working-ledger.sqlite');self.memory=False
        else:
            self.conn=sqlite3.connect(':memory:')
        self.configure()
        if compact:
            source.backup(self.conn,pages=16384)
            self.configure()
        else:
            self.conn.executescript('''
                CREATE TABLE names(id INTEGER PRIMARY KEY, value TEXT UNIQUE NOT NULL);
                CREATE TABLE allocations(sha BLOB PRIMARY KEY, row_sha BLOB NOT NULL,
                    leaf INTEGER NOT NULL, owner INTEGER NOT NULL) WITHOUT ROWID;
                CREATE TABLE drained(index_sha BLOB PRIMARY KEY, bin_sha BLOB NOT NULL) WITHOUT ROWID;
                CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
                CREATE VIEW selected AS SELECT lower(hex(a.sha)) AS text_sha,
                    lower(hex(a.row_sha)) AS row_key_sha256,l.value AS leaf,o.value AS allocation
                    FROM allocations a JOIN names l ON l.id=a.leaf JOIN names o ON o.id=a.owner;
            ''')
        self.names={value:n for n,value in self.conn.execute('SELECT id,value FROM names')}
        self.values={n:value for value,n in self.names.items()}
        self.count=int(self.conn.execute("SELECT value FROM metadata WHERE name='count'").fetchone()[0]) if compact else 0
        if source and not compact:
            cols={r[1]for r in source.execute('PRAGMA table_info(selected)')}
            owner='allocation' if 'allocation' in cols else "'prior'"
            cursor=source.execute('SELECT text_sha,row_key,leaf,'+owner+' FROM selected')
            while True:
                rows=cursor.fetchmany(100000)
                if not rows:break
                batch=[(key(sha),row_key_hash(row),self.name(leaf),self.name(allocation))
                       for sha,row,leaf,allocation in rows]
                self.conn.executemany('INSERT INTO allocations VALUES(?,?,?,?)',sorted(batch))
                self.count+=len(batch);self.commit()
            if source.execute("SELECT 1 FROM sqlite_master WHERE name='drained_outputs'").fetchone():
                self.conn.executemany('INSERT INTO drained VALUES(?,?)',
                    ((key(a),key(b))for a,b in source.execute('SELECT index_sha,bin_sha FROM drained_outputs')))
        if source:source.close()
        self.commit()
        if use_seen and self.count*120<=self.seen_limit:
            self.seen={r[0]for r in self.conn.execute('SELECT sha FROM allocations')}
            self.bound_seen()

    def configure(self):
        self.conn.execute('PRAGMA synchronous=OFF')
        self.conn.execute('PRAGMA journal_mode=MEMORY')
        self.conn.execute('PRAGMA temp_store=MEMORY')
        self.conn.execute('PRAGMA locking_mode=EXCLUSIVE')
        self.conn.execute('PRAGMA cache_size=-'+str(self.cache_mib*1024))

    def name(self, value):
        if value not in self.names:
            cursor=self.conn.execute('INSERT INTO names(value) VALUES(?)',(value,))
            n=cursor.lastrowid;self.names[value]=n;self.values[n]=value
        return self.names[value]

    def bound_seen(self):
        if self.seen is not None and len(self.seen)*65+sys.getsizeof(self.seen)>self.seen_limit:
            self.seen=None

    def contains_many(self, hashes):
        if self.seen is not None:return {h for h in hashes if h in self.seen}
        result=set()
        for start in range(0,len(hashes),900):
            group=hashes[start:start+900]
            if group:
                result.update(r[0]for r in self.conn.execute(
                    'SELECT sha FROM allocations WHERE sha IN ('+','.join('?'for _ in group)+')',group))
        return result

    def get(self, sha):
        row=self.conn.execute('SELECT row_sha,leaf,owner FROM allocations WHERE sha=?',(key(sha),)).fetchone()
        return None if row is None else (row[0],self.values[row[1]],self.values[row[2]])

    def get_many(self, hashes):
        result={}
        for start in range(0,len(hashes),900):
            group=hashes[start:start+900]
            if group:
                for sha,row,leaf,owner in self.conn.execute(
                    'SELECT sha,row_sha,leaf,owner FROM allocations WHERE sha IN ('+','.join('?'for _ in group)+')',group):
                    result[sha]=(row,self.values[leaf],self.values[owner])
        return result

    def add_many(self, rows, leaf, owner):
        l,o=self.name(leaf),self.name(owner)
        for start in range(0,len(rows),100000):
            group=rows[start:start+100000]
            batch=sorted((r.get('_sha') or key(r['source_text_sha256']),
                r.get('_row_sha') or row_key_hash(r['row_key']),l,o)for r in group)
            if self.bulk_insert_rows:
                for i in range(0,len(batch),self.bulk_insert_rows):
                    values=batch[i:i+self.bulk_insert_rows]
                    self.conn.execute('INSERT INTO allocations VALUES '+','.join('(?,?,?,?)'for _ in values),
                        tuple(chain.from_iterable(values)))
            else:self.conn.executemany('INSERT INTO allocations VALUES(?,?,?,?)',batch)
            self.count+=len(batch)
            if self.seen is not None:self.seen.update(r[0]for r in batch)
            self.uncommitted_rows+=len(batch)
            if not self.commit_rows or self.uncommitted_rows>=self.commit_rows:self.commit()
        self.bound_seen()

    def change_owner(self, sha, owner):
        self.conn.execute('UPDATE allocations SET owner=? WHERE sha=?',(self.name(owner),key(sha)))

    def drained(self, index_sha, bin_sha):
        row=self.conn.execute('SELECT bin_sha FROM drained WHERE index_sha=?',(key(index_sha),)).fetchone()
        if row and row[0]!=key(bin_sha):raise ValueError('drained index token identity differs')
        return bool(row)

    def mark_drained(self, index_sha, bin_sha):
        self.conn.execute('INSERT OR IGNORE INTO drained VALUES(?,?)',(key(index_sha),key(bin_sha)))

    def commit(self):
        self.conn.execute("INSERT OR REPLACE INTO metadata VALUES('schema','ci1_compact_ledger_v2')")
        self.conn.execute("INSERT OR REPLACE INTO metadata VALUES('count',?)",(str(self.count),))
        self.conn.commit()
        self.uncommitted_rows=0
        pages=self.conn.execute('PRAGMA page_count').fetchone()[0]
        size=pages*self.conn.execute('PRAGMA page_size').fetchone()[0]
        if self.memory and size>self.memory_limit:
            target=sqlite3.connect(self.work/'working-ledger.sqlite')
            target.execute('PRAGMA synchronous=OFF');target.execute('PRAGMA journal_mode=OFF')
            self.conn.backup(target,pages=16384)
            self.conn.close();self.conn=target;self.memory=False;self.configure()

    def snapshot(self, destination):
        self.commit();destination=Path(destination);temporary=destination.with_suffix('.sqlite.tmp')
        temporary.unlink(missing_ok=True)
        target=sqlite3.connect(temporary)
        target.execute('PRAGMA synchronous=OFF');target.execute('PRAGMA journal_mode=OFF')
        self.conn.backup(target,pages=16384)
        if target.execute('PRAGMA quick_check').fetchone()[0]!='ok':
            raise ValueError('invalid completed ledger snapshot')
        if target.execute('SELECT COUNT(*) FROM allocations').fetchone()[0]!=self.count:
            raise ValueError('completed ledger count differs')
        target.close();temporary.replace(destination)
        return destination

    def close(self):
        self.conn.close();self.seen=None
