"""Native, hash-partitioned reservation engine and exact quota coordinator.

The catalogue must be built from verified conversion indexes, frozen heldout
exclusions and an immutable predecessor ownership snapshot. This module is
CPU tooling, not a deployed replacement for the preparation dispatcher.
Each partition has one writer; all phase receipts must reconcile before any
physical materializer consumes them. DuckDB 1.4.2 executes bulk selection.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from production.compact_ledger import key
from production.document_packing import file_sha
from production.packing_worker import fingerprint

SCHEMA = 'ci1_partitioned_allocation_v1'


def partition_for(sha, partitions):
    if type(partitions) is not int or not 1 <= partitions <= 64:
        raise ValueError('invalid allocation partition count')
    return int.from_bytes(key(sha)[:4], 'big') % partitions


def capacity_split(target, capacities):
    """Exact proportional integers, including empty/unequal hash partitions.

    Never assume each worker has 20% of a scarce dataset. Capacity must reflect
    the current phase, after protected records and duplicate exclusion.
    """
    if (type(target) is not int or target < 0 or not capacities
            or any(type(n) is not int or n < 0 for n in capacities.values())):
        raise ValueError('invalid capacity or target')
    total = sum(capacities.values())
    if target > total:
        raise ValueError('eligible capacity is below the approved quota')
    if not total:
        return dict.fromkeys(capacities, 0)
    allocated = {k: target*n//total for k, n in capacities.items()}
    order = sorted(capacities, key=lambda k: (-(target*capacities[k] % total), k))
    for k in order[:target-sum(allocated.values())]:
        allocated[k] += 1
    if sum(allocated.values()) != target or any(allocated[k] > capacities[k] for k in capacities):
        raise ValueError('capacity split failed exact reconciliation')
    return allocated


def phase_targets(quotas, capacities):
    """quotas map (leaf, child stratum) to tokens; capacities map worker to map."""
    if not capacities:
        raise ValueError('missing partition census')
    result = {worker: {} for worker in capacities}
    for quota, target in quotas.items():
        split = capacity_split(target, {w: c.get(quota, 0) for w, c in capacities.items()})
        for worker, n in split.items():
            if n:
                result[worker][quota] = n
    return result


def catalogue_record(record, *, leaf, stratum, source_order, exclusions,
                     previously_allocated=False):
    """Build a typed row only after the caller verifies its index/bin receipt.

    Prior ownership includes whole protected records. Their quota-cut tails
    cannot enter this fresh catalogue. Existing dense suffixes need a separate
    explicitly bound handoff, not a fresh full-record entry.
    """
    if previously_allocated or exclusions.excludes(record):
        return None
    for field in ('source_text_sha256', 'source_bin_sha256', 'source_index_sha256'):
        key(record[field])
    if (type(record['n_tokens']) is not int or record['n_tokens'] <= 0
            or type(record['token_start']) is not int or record['token_start'] < 0
            or not all(isinstance(x, str) and x for x in
                       (leaf, stratum, source_order, record['row_key']))):
        raise ValueError('invalid catalogue source coordinates')
    document = {k: record[k] for k in ('doc_id', 'row_key', 'split_family_id',
        'source_text_sha256', 'source_bin_uri', 'source_bin_sha256', 'source_index_sha256')}
    document.update(source_token_start=record['token_start'], original_content_tokens=record['n_tokens'])
    return {'sha': record['source_text_sha256'], 'leaf': leaf, 'stratum': stratum,
        'ordering': source_order, 'n': record['n_tokens'], 'payload': json.dumps(document,
            ensure_ascii=False, separators=(',', ':'))}


def compile_catalogues(sources, prior_hashes, output, *, source_priority,
                       partitions, predecessor_sha256, exclusions_sha256,
                       memory_limit='8GB', threads=4):
    """Canonicalize globally in native SQL, then write disjoint Parquet parts.

    sources are already heldout-filtered catalogue rows from verified source
    receipts. prior_hashes is a complete Parquet SHA census exported from the
    pinned warmup/protected ledger. Source priority is an explicit data policy:
    if duplicate text occurs in two quotas, this decides which retains it.
    Capacity shortfalls remain errors at phase planning, never silent fallback.
    """
    import duckdb
    if duckdb.__version__ != '1.4.2':
        raise ValueError('allocation engine requires DuckDB 1.4.2')
    partition_for('00'*32, partitions)
    key(predecessor_sha256);key(exclusions_sha256)
    if not sources or len(source_priority) != len(set(source_priority)) or not source_priority:
        raise ValueError('missing or ambiguous source priority policy')
    output = Path(output)
    if output.exists():
        raise ValueError('existing catalogue directory requires explicit recovery')
    output.mkdir(parents=True)
    conn = duckdb.connect(str(output/'compiler.duckdb'), config={'memory_limit': memory_limit, 'threads': threads})
    try:
        for spec in list(sources)+list(prior_hashes):
            path = Path(spec['path'])
            if path.stat().st_size != spec['bytes'] or file_sha(path) != spec['sha256']:
                raise ValueError('input catalogue or prior census identity differs')
        conn.execute('CREATE TABLE priority(leaf VARCHAR PRIMARY KEY, rank INTEGER)')
        conn.executemany('INSERT INTO priority VALUES(?,?)', [(leaf,i)for i,leaf in enumerate(source_priority)])
        conn.execute('CREATE TABLE prior(sha VARCHAR PRIMARY KEY)')
        for spec in prior_hashes:
            conn.execute('INSERT INTO prior SELECT sha FROM read_parquet(?)', [spec['path']])
        # A zero-row predecessor census is still an explicit input artifact;
        # callers may not accidentally omit existing warmup/QAT ownership.
        if not prior_hashes:
            raise ValueError('missing immutable predecessor SHA census')
        source_paths=[str(spec['path'])for spec in sources]
        conn.execute('CREATE TABLE raw AS SELECT * FROM read_parquet(?)', [source_paths])
        invalid=conn.execute("""SELECT COUNT(*) FROM raw WHERE sha IS NULL OR
            NOT regexp_full_match(sha,'[0-9a-f]{64}') OR n IS NULL OR n<=0 OR
            ordering IS NULL OR ordering='' OR stratum IS NULL OR stratum='' OR payload IS NULL""").fetchone()[0]
        invalid_prior=conn.execute("SELECT COUNT(*) FROM prior WHERE NOT regexp_full_match(sha,'[0-9a-f]{64}')").fetchone()[0]
        absent=conn.execute('SELECT COUNT(*) FROM raw ANTI JOIN priority USING(leaf)').fetchone()[0]
        if invalid or invalid_prior or absent:
            raise ValueError('invalid catalogue rows or missing source priority')
        if conn.execute('SELECT 1 FROM raw GROUP BY sha HAVING MIN(n) != MAX(n) LIMIT 1').fetchone():
            raise ValueError('identical text SHA has incompatible tokenizer lengths')
        conn.execute('''CREATE TABLE canonical AS
            SELECT sha,leaf,stratum,ordering,n,payload FROM (
              SELECT raw.*,ROW_NUMBER() OVER (PARTITION BY sha ORDER BY priority.rank,ordering,payload) AS duplicate_rank
              FROM raw JOIN priority USING(leaf) ANTI JOIN prior USING(sha))
            WHERE duplicate_rank=1''')
        parts=[]
        for partition in range(partitions):
            path=output/f'part-{partition:02d}.parquet'
            conn.execute("COPY (SELECT * FROM canonical WHERE (('0x'||substr(sha,1,8))::UBIGINT % $partition_count)=$partition_id ORDER BY leaf,stratum,ordering,sha) TO $destination (FORMAT PARQUET, COMPRESSION ZSTD)",
                {'partition_count':partitions,'partition_id':partition,'destination':str(path)})
            parts.append({'partition':partition,'path':str(path),'bytes':path.stat().st_size,'sha256':file_sha(path)})
        counts=conn.execute('SELECT (SELECT COUNT(*) FROM raw),(SELECT COUNT(*) FROM prior),(SELECT COUNT(*) FROM canonical),(SELECT COALESCE(SUM(n),0) FROM canonical)').fetchone()
        result={'schema':SCHEMA,'partitions':partitions,'predecessor_sha256':predecessor_sha256,
            'exclusions_sha256':exclusions_sha256,'source_priority':source_priority,
            'input_sha256':[spec['sha256']for spec in sources],
            'prior_census_sha256':[spec['sha256']for spec in prior_hashes],
            'input_records':counts[0],'prior_records':counts[1],'eligible_unique_records':counts[2],
            'eligible_unique_content_tokens':counts[3],'catalogues':parts,
            'physical_creation_finished':False,'native_training_accepted':False}
        (output/'CATALOGUE.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:
        conn.close()


class PartitionAllocator:
    """One bounded native database per hash partition, reusable across phases.

    Source-priority canonicalization MUST occur across all leaves before
    splitting a catalogue. One SHA may not belong to two source quotas. The
    engine rejects unresolved duplicates instead of silently changing quotas.
    """
    def __init__(self, database, catalogues, *, partition, partitions,
                 predecessor_sha256, exclusions_sha256, memory_limit='8GB', threads=4):
        import duckdb
        if duckdb.__version__ != '1.4.2':
            raise ValueError('allocation engine requires DuckDB 1.4.2')
        partition_for('00'*32, partitions)
        if type(partition) is not int or not 0 <= partition < partitions:
            raise ValueError('invalid partition identity')
        key(predecessor_sha256); key(exclusions_sha256)
        self.path = Path(database)
        if self.path.exists():
            raise ValueError('existing allocation database requires explicit recovery')
        self.partition, self.partitions = partition, partitions
        self.predecessor = predecessor_sha256
        self.exclusions = exclusions_sha256
        self.catalogues = []
        self.conn = duckdb.connect(str(self.path), config={'memory_limit': memory_limit, 'threads': threads})
        self.conn.execute('CREATE TABLE candidates(sha VARCHAR PRIMARY KEY, leaf VARCHAR, stratum VARCHAR, ordering VARCHAR, n BIGINT, payload JSON, consumed BIGINT DEFAULT 0)')
        self.conn.execute('CREATE TABLE phases(id VARCHAR PRIMARY KEY, receipt_sha256 VARCHAR)')
        try:
            for spec in catalogues:
                path = Path(spec['path'])
                if path.stat().st_size != spec['bytes'] or file_sha(path) != spec['sha256']:
                    raise ValueError('catalogue identity differs')
                self.catalogues.append(spec['sha256'])
                # Parquet columns come from the reviewed catalogue compiler;
                # no Python JSON/row loop is on the native reservation path.
                self.conn.execute('INSERT INTO candidates(sha,leaf,stratum,ordering,n,payload) SELECT sha,leaf,stratum,ordering,n,payload::JSON FROM read_parquet(?)', [str(path)])
            invalid = self.conn.execute("""SELECT COUNT(*) FROM candidates WHERE
                sha IS NULL OR NOT regexp_full_match(sha,'[0-9a-f]{64}') OR
                leaf IS NULL OR leaf='' OR stratum IS NULL OR stratum='' OR
                ordering IS NULL OR ordering='' OR n IS NULL OR n<=0 OR payload IS NULL OR
                json_extract_string(payload,'$.source_text_sha256') IS DISTINCT FROM sha OR
                try_cast(json_extract_string(payload,'$.original_content_tokens') AS BIGINT) IS DISTINCT FROM n OR
                (('0x'||substr(sha,1,8))::UBIGINT % ?) != ?""", [partitions, partition]).fetchone()[0]
            if invalid:
                raise ValueError('catalogue contains invalid or wrongly owned records')
        except BaseException:
            self.conn.close()
            self.path.unlink(missing_ok=True)
            raise

    def capacities(self):
        return {(leaf, stratum): int(n) for leaf, stratum, n in self.conn.execute(
            'SELECT leaf,stratum,SUM(n-consumed) FROM candidates WHERE consumed<n GROUP BY leaf,stratum').fetchall()}

    def reserve(self, phase, targets, output, *, reuse_suffixes=False):
        if not isinstance(phase, str) or not phase or self.conn.execute('SELECT 1 FROM phases WHERE id=?', [phase]).fetchone():
            raise ValueError('phase has already been reserved or is invalid')
        if any(type(n) is not int or n <= 0 for n in targets.values()):
            raise ValueError('invalid partition phase quotas')
        capacities = self.capacities()
        if any(n > capacities.get(k, 0) for k, n in targets.items()):
            raise ValueError('partition capacity is below its exact phase quota')
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name+'.partial')
        if output.exists() or temporary.exists():
            raise ValueError('reservation output already exists')
        self.conn.execute('BEGIN TRANSACTION')
        committed = False
        try:
            self.conn.execute('CREATE TEMP TABLE chosen(sha VARCHAR, leaf VARCHAR, stratum VARCHAR, document_offset BIGINT, take BIGINT, payload JSON)')
            for (leaf, stratum), target in sorted(targets.items()):
                self.conn.execute('''INSERT INTO chosen
                    WITH ranked AS (
                        SELECT *, COALESCE(SUM(n-consumed) OVER (
                          ORDER BY ordering,sha ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS prefix
                        FROM candidates WHERE leaf=? AND stratum=? AND consumed<n)
                    SELECT sha,leaf,stratum,consumed,LEAST(n-consumed,?-prefix)::BIGINT,payload
                    FROM ranked WHERE prefix<?''', [leaf, stratum, target, target])
            observed = {(leaf, stratum): int(n) for leaf, stratum, n in self.conn.execute(
                'SELECT leaf,stratum,SUM(take) FROM chosen GROUP BY leaf,stratum').fetchall()}
            if observed != targets:
                raise ValueError('native reservation counters differ from assigned quotas')
            # Output uses existing fixed-materializer record semantics.
            self.conn.execute('''COPY (SELECT json_merge_patch(payload,
                to_json(struct_pack(leaf := leaf, stratum := stratum,
                    allocation := $phase, allocated_document_offset := document_offset,
                    allocated_content_tokens := take, whole_record_protected := true))) AS record
                FROM chosen ORDER BY leaf,stratum,sha)
                TO $destination (FORMAT CSV, DELIMITER '\t', HEADER false, QUOTE '', ESCAPE '')''',
                {'phase':phase,'destination':str(temporary)})
            # A single unquoted JSON column gives newline-delimited records,
            # including escaped newlines/tabs inside source coordinates.
            if reuse_suffixes:
                self.conn.execute('UPDATE candidates SET consumed=chosen.document_offset+chosen.take FROM chosen WHERE candidates.sha=chosen.sha')
            else:
                self.conn.execute('UPDATE candidates SET consumed=n FROM chosen WHERE candidates.sha=chosen.sha')
            counts = self.conn.execute('SELECT COUNT(*),COALESCE(SUM(take),0) FROM chosen').fetchone()
            receipt = {'schema': SCHEMA, 'phase': phase, 'partition': self.partition,
                'partitions': self.partitions, 'predecessor_sha256': self.predecessor,
                'exclusions_sha256': self.exclusions, 'catalogue_sha256': sorted(self.catalogues),
                'source_targets': [{'leaf': l, 'stratum': s, 'tokens': n} for (l,s),n in sorted(observed.items())],
                'records': counts[0], 'content_tokens': counts[1], 'reuse_suffixes': bool(reuse_suffixes),
                'selection': {'path': str(output), 'bytes': temporary.stat().st_size, 'sha256': file_sha(temporary)},
                'physical_creation_finished': False, 'native_training_accepted': False}
            digest = fingerprint(receipt)
            self.conn.execute('INSERT INTO phases VALUES(?,?)', [phase, digest])
            self.conn.execute('DROP TABLE chosen')
            self.conn.execute('COMMIT')
            committed = True
            temporary.rename(output)
            self.predecessor = digest
            return receipt
        except BaseException:
            if not committed:
                self.conn.execute('ROLLBACK')
            temporary.unlink(missing_ok=True)
            raise

    def close(self):
        self.conn.close()


def reconcile_phase(receipts, *, phase, targets, partitions, exclusions_sha256,
                    predecessor_sha256):
    """Barrier before materialization: all partitions, exact child quotas, hashes.

    predecessor_sha256 maps partition IDs to their expected previous receipt
    (or the shared imported ledger). These are immutable DAG dependencies.
    """
    if len(receipts) != partitions or {r['partition'] for r in receipts} != set(range(partitions)):
        raise ValueError('missing, duplicate or foreign partition receipts')
    observed = defaultdict(int)
    import duckdb
    if duckdb.__version__ != '1.4.2':
        raise ValueError('allocation engine requires DuckDB 1.4.2')
    for r in receipts:
        if (r['schema'] != SCHEMA or r['phase'] != phase or r['partitions'] != partitions
                or r['exclusions_sha256'] != exclusions_sha256
                or r['predecessor_sha256'] != predecessor_sha256[r['partition']]):
            raise ValueError('partition receipt dependency differs')
        local = {}
        for row in r['source_targets']:
            quota = (row['leaf'], row['stratum'])
            if quota in local or type(row['tokens']) is not int or row['tokens'] <= 0:
                raise ValueError('invalid partition receipt quota')
            local[quota] = row['tokens']
            observed[quota] += row['tokens']
        if sum(local.values()) != r['content_tokens']:
            raise ValueError('partition total differs from child quotas')
        spec = r['selection']
        if Path(spec['path']).stat().st_size != spec['bytes'] or file_sha(spec['path']) != spec['sha256']:
            raise ValueError('partition reservation payload differs')
        # Verify counters against actual records, not just a self-consistent
        # worker receipt. Native JSON scanning stays bounded by engine memory.
        with duckdb.connect(config={'memory_limit':'2GB','threads':2}) as conn:
            columns = {'source_text_sha256':'VARCHAR','leaf':'VARCHAR','stratum':'VARCHAR',
                'allocated_content_tokens':'BIGINT','allocated_document_offset':'BIGINT',
                'original_content_tokens':'BIGINT','allocation':'VARCHAR','whole_record_protected':'BOOLEAN'}
            if spec['bytes']:
                conn.execute("CREATE TEMP TABLE selection AS SELECT * FROM read_json(?,format='newline_delimited',columns=?)",
                    [spec['path'],columns])
                invalid=conn.execute("""SELECT COUNT(*) FROM selection WHERE
                    source_text_sha256 IS NULL OR NOT regexp_full_match(source_text_sha256,'[0-9a-f]{64}') OR
                    (('0x'||substr(source_text_sha256,1,8))::UBIGINT % ?) != ? OR
                    allocated_content_tokens IS NULL OR allocated_content_tokens<=0 OR
                    allocated_document_offset IS NULL OR allocated_document_offset<0 OR
                    original_content_tokens IS NULL OR allocated_document_offset+allocated_content_tokens>original_content_tokens OR
                    allocation IS DISTINCT FROM ? OR whole_record_protected IS DISTINCT FROM true""",
                    [partitions,r['partition'],phase]).fetchone()[0]
                n,unique=conn.execute('SELECT COUNT(*),COUNT(DISTINCT source_text_sha256) FROM selection').fetchone()
                actual={(l,s):int(n)for l,s,n in conn.execute('SELECT leaf,stratum,SUM(allocated_content_tokens) FROM selection GROUP BY leaf,stratum').fetchall()}
                if invalid or n!=unique or n!=r['records'] or actual!=local:
                    raise ValueError('reservation records differ from partition ownership or quotas')
            elif r['records'] or local:
                raise ValueError('empty reservation has nonzero counters')
    if dict(observed) != targets:
        raise ValueError('parallel phase changes approved source/child quotas')
    return {'schema': SCHEMA, 'phase': phase, 'all_curriculum_quotas_satisfied': True,
        'content_tokens': sum(observed.values()), 'partition_receipt_sha256': [fingerprint(r) for r in sorted(receipts,key=lambda r:r['partition'])],
        'physical_creation_finished': False, 'native_training_accepted': False}


def main():
    """Execute external local-file plans; does not launch cloud machines."""
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['compile','reserve'])
    parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    if args.operation=='compile':
        result=compile_catalogues(plan['sources'],plan['prior_hashes'],plan['output'],
            source_priority=plan['source_priority'],partitions=plan['partitions'],
            predecessor_sha256=plan['predecessor_sha256'],exclusions_sha256=plan['exclusions_sha256'],
            memory_limit=plan.get('memory_limit','8GB'),threads=plan.get('threads',4))
        print(json.dumps(result,sort_keys=True))
    else:
        engine=PartitionAllocator(plan['database'],plan['catalogues'],partition=plan['partition'],
            partitions=plan['partitions'],predecessor_sha256=plan['predecessor_sha256'],
            exclusions_sha256=plan['exclusions_sha256'],memory_limit=plan.get('memory_limit','8GB'),
            threads=plan.get('threads',4))
        try:
            for phase in plan['phases']:
                targets={(r['leaf'],r['stratum']):r['tokens']for r in phase['source_targets']}
                if len(targets)!=len(phase['source_targets']):
                    raise ValueError('duplicate phase source/stratum quota')
                result=engine.reserve(phase['id'],targets,phase['output'],reuse_suffixes=phase.get('reuse_suffixes',False))
                Path(phase['receipt']).write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(result,sort_keys=True),flush=True)
        finally:
            engine.close()


if __name__=='__main__':main()
