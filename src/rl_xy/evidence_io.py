"""Lossless streaming evidence encodings and bounded new-runner phases.

Canonical semantic SHA means SHA-256 of compact sorted-key UTF-8 JSON, with
no trailing newline, not a hash of whitespace-dependent pretty JSON. Gzip is
an encoding of those exact canonical bytes; it never replaces native logs.
"""
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, fields
from functools import lru_cache
import gzip
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading


def _default(value):
    if isinstance(value, Path): return str(value)
    if is_dataclass(value): return asdict(value)
    raise TypeError('cannot encode '+type(value).__name__)


def _canonical_parts(value, encoder, depth=0):
    """Stream three plain-container levels, then C-encode one bounded leaf.

    For a unified resource contract depth0 is global, depth1 is ranks,
    depth2 is one rank, and depth3 is that rank's nodes. Thus a C leaf never
    materializes the whole global contract. Integer-key/custom containers keep
    the original encoder fallback and its key/type/error semantics.
    """
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            yield from encoder.iterencode(value)
            return
        if depth < 3:
            yield '{'
            for index,key in enumerate(sorted(value)):
                if index:yield ','
                yield from encoder.iterencode(key,_one_shot=True)
                yield ':'
                yield from _canonical_parts(value[key],encoder,depth+1)
            yield '}'
            return
    elif type(value) in (list,tuple) and depth < 3:
        yield '['
        for index,item in enumerate(value):
            if index:yield ','
            yield from _canonical_parts(item,encoder,depth+1)
        yield ']'
        return
    elif type(value) not in (list,tuple,str,int,float,bool,type(None)):
        yield from encoder.iterencode(value)
        return
    yield from encoder.iterencode(value,_one_shot=True)


def _canonical_chunks(value, *, buffer_bytes=65536):
    """Bound prefix buffering; C fast path materializes only depth3 leaves."""
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False, default=_default)
    pending = [];size = 0
    for part in _canonical_parts(value,encoder):
        piece = part.encode('utf-8');pending.append(piece);size += len(piece)
        if size >= buffer_bytes:
            yield b''.join(pending);pending = [];size = 0
    if pending: yield b''.join(pending)


def canonical_semantic_sha256(value):
    h = hashlib.sha256()
    for chunk in _canonical_chunks(value): h.update(chunk)
    return h.hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def _write_json(path, value, *, compressed):
    target = Path(path);temporary = target.with_name('.'+target.name+'.tmp')
    if compressed and target.suffix != '.gz':
        raise ValueError('gzip JSON evidence must use an explicit .gz suffix')
    canonical = hashlib.sha256();uncompressed_bytes = 0
    try:
        with temporary.open('wb') as raw:
            if compressed:
                stream = gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0)
            else:
                stream = raw
            try:
                for chunk in _canonical_chunks(value):
                    canonical.update(chunk);uncompressed_bytes += len(chunk);stream.write(chunk)
            finally:
                if compressed: stream.close()
        os.replace(temporary, target)
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass
    digest = canonical.hexdigest()
    return dict(encoding='gzip_json' if compressed else 'minified_json',
        canonical_semantic_sha256=digest, uncompressed_canonical_sha256=digest,
        uncompressed_canonical_bytes=uncompressed_bytes,
        stored_sha256=_sha256_file(target), stored_bytes=target.stat().st_size)


def write_minified_json(path, value):
    return _write_json(path, value, compressed=False)


def write_gzip_json(path, value):
    return _write_json(path, value, compressed=True)


def load_json_auto(path):
    """Read explicit .json or .json.gz, without creating an expanded copy."""
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream: return json.load(stream)


class _HashedTextSink:
    """UTF-8 text sink with no newline translation and bounded gzip writes."""
    def __init__(self, stream):
        self.stream=stream;self.sha=hashlib.sha256();self.size=0;self.pending=bytearray()
    def write(self, text):
        chunk=text.encode('utf-8');self.sha.update(chunk);self.size+=len(chunk);self.pending.extend(chunk)
        if len(self.pending)>=65536:self.flush()
        return len(text)
    def flush(self):
        if self.pending:self.stream.write(self.pending);self.pending.clear()


def _write_gzip_table(path, write_rows, *, encoding):
    target=Path(path);temporary=target.with_name('.'+target.name+'.tmp')
    if target.suffix!='.gz':raise ValueError('gzip table evidence must use an explicit .gz suffix')
    try:
        with temporary.open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0) as stream:
                sink=_HashedTextSink(stream);count=write_rows(sink);sink.flush()
        os.replace(temporary,target)
    finally:
        try:temporary.unlink()
        except FileNotFoundError:pass
    return dict(encoding=encoding,uncompressed_sha256=sink.sha.hexdigest(),
        uncompressed_bytes=sink.size,stored_sha256=_sha256_file(target),
        stored_bytes=target.stat().st_size,row_count=count)


@lru_cache(maxsize=64)
def _row_field_names(kind):
    return tuple(item.name for item in fields(kind))


def _readonly_row(value):
    """A fresh field mapping without copying large read-only metadata/tapes."""
    if is_dataclass(value):
        if isinstance(value,type):
            # Preserve asdict's rejection of dataclass classes as row values.
            return asdict(value)
        return {name:getattr(value,name) for name in _row_field_names(type(value))}
    return dict(value)


def _csv_dataclass_default(value):
    # Legacy CSV first ran asdict on a dataclass row, recursively converting
    # nested dataclasses. Do the equivalent conversion only when encountered
    # during JSON serialization. Path nested inside a dict still fails exactly
    # as legacy json.dumps without a custom Path default did.
    if is_dataclass(value):return asdict(value)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def _write_gzip_jsonl(path,rows):
    def write(stream):
        count=0
        for row in rows:
            value=_readonly_row(row)
            # These are the exact legacy per-row spaces, ordering and newline;
            # canonical compact JSON is deliberately NOT used for table bytes.
            stream.write(json.dumps(value,ensure_ascii=False,sort_keys=True,default=_default)+'\n');count+=1
        return count
    return _write_gzip_table(path,write,encoding='gzip_jsonl')


def _write_gzip_csv(path,rows,fieldnames):
    def write(stream):
        writer=csv.DictWriter(stream,fieldnames=fieldnames,extrasaction='raise');writer.writeheader();count=0
        for value in rows:
            dataclass_row=is_dataclass(value);row=_readonly_row(value)
            for key,item in list(row.items()):
                if dataclass_row and is_dataclass(item):item=asdict(item)
                if isinstance(item,(dict,list,tuple)):
                    options={'default':_csv_dataclass_default} if dataclass_row else {}
                    row[key]=json.dumps(item,ensure_ascii=False,sort_keys=True,**options)
                elif isinstance(item,Path):row[key]=str(item)
            writer.writerow(row);count+=1
        return count
    return _write_gzip_table(path,write,encoding='gzip_csv')


def write_aligned_tables(output_dir,*,operations,events,summaries,objects,snapshots,
        transfers,checks,parameters=()):
    """Write lossless gzip tables for the new runner only, row by row.

    Decompressed bytes match reporting.write_tables, including JSONL spaces
    and CSV CRLF/quoting. Source iterables are consumed once, without full list
    copies. Native Astra stdout/debug/resource files are never touched.
    """
    from .reporting import write_csv,write_json
    from .types import AstraRankSummary,ObjectRecord,MemorySnapshot,TransferRecord,CheckResult,ParameterRow
    output_dir=Path(output_dir);records={}
    for name,rows in [('operations.jsonl',operations),('events.jsonl',events)]:
        path=output_dir/(name+'.gz');records[name]=dict(path=path.name,**_write_gzip_jsonl(path,rows))
    for name,rows,kind in [('objects.csv',objects,ObjectRecord),('memory.csv',snapshots,MemorySnapshot),
            ('transfers.csv',transfers,TransferRecord)]:
        path=output_dir/(name+'.gz')
        records[name]=dict(path=path.name,**_write_gzip_csv(path,rows,[item.name for item in fields(kind)]))
    for name,rows,kind in [('astra_rank_summary.csv',summaries,AstraRankSummary),
            ('checks.csv',checks,CheckResult),('parameters.csv',parameters,ParameterRow)]:
        values=list(rows);path=output_dir/name
        write_csv(path,values,[item.name for item in fields(kind)])
        digest=_sha256_file(path);size=path.stat().st_size
        records[name]=dict(path=name,encoding='csv',stored_sha256=digest,stored_bytes=size,
            uncompressed_sha256=digest,uncompressed_bytes=size,row_count=len(values))
    result=dict(schema_version='rlxy.aligned_tables/1',tables=records,
        encoding_only_change=True,uncompressed_format='exact reporting.write_tables UTF8 bytes',
        native_raw_logs_preserved=True,
        reconstruction='gzip.open(path, rt, encoding=UTF8, newline=empty); compare decompressed byte SHA before parsing',
        original_plain_large_tables_written=False)
    write_json(output_dir/'table_format.json',result)
    return result


class EvidencePhaseTimeout(TimeoutError):
    """A new aligned-runner phase exhausted its admitted wall-clock budget."""


@contextmanager
def phase_deadline(phase, seconds, *, enabled):
    """Interrupt one main-thread phase and always restore SIGALRM ownership.

    The aligned runner enables this only inside Linux Docker. Existing real
    interval timers fail closed: their handlers/deadlines are never borrowed,
    suspended or silently replaced. Native execution has its own subprocess
    timeout and is deliberately outside these phase contexts.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive finite phase deadline required')
    if not enabled:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('phase deadline requires the main thread')
    if not hasattr(signal, 'setitimer'):
        raise ValueError('real interval timer unavailable')
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise ValueError('preexisting real timer cannot be overridden by phase deadline')
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise EvidencePhaseTimeout(f'{phase} phase exceeded {seconds:g}s hard deadline')
    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
