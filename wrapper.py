#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DBA bandits index-selection wrapper (long-lived HTTP server).

Implements the C3UCB combinatorial contextual bandit (Perera, Oetomo,
Rubinstein, and Borovica-Gajic, ICDE 2021) behind the index-selection HTTP
protocol: GET /health, POST /recommend, GET /state, POST /shutdown.

Each round the bandit (ridge-regression confidence bounds over per-arm
context vectors built from predicate/payload columns) selects a full index
configuration via the greedy oracle under a storage budget. The configuration
is evaluated with cost-only what-if (EXPLAIN under hypothetical indexes), and
per-arm rewards (table-access cost reduction minus creation cost) update the
bandit. Learning state (matrices, arm store, query store) persists in memory
and on disk, so recommendations improve as more workloads are observed.

All what-if analysis uses HypoPG session-local hypothetical indexes on
--eval-dsn (defaults to --dsn); only --apply executes real CREATE INDEX on
--dsn. Workload predicates/payload are derived from raw SQL with sqlglot.
Online training (extra what-if rounds on the eval DSN) runs in a background
subprocess (train_trigger) and never blocks the recommendation.
"""
import sys
import os
import json
import re
import time
import hashlib
import threading
import traceback
import warnings
import logging
import tempfile
import subprocess
import pickle
import copy
import random
from collections import deque

warnings.filterwarnings("ignore")

REPO = os.environ.get("DBABANDITS_REPO", os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Library logs go to a file; stdout stays clean.
_STATE_DIR_ENV = os.environ.get("DBABANDITS_STATE_DIR", os.path.join(REPO, "skill_state"))
os.makedirs(_STATE_DIR_ENV, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(_STATE_DIR_ENV, "source.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

import numpy  # noqa: E402

# numpy >= 1.25 removed asscalar(), which the algorithm uses.
if not hasattr(numpy, "asscalar"):
    numpy.asscalar = lambda a: a.item()

import constants  # noqa: E402

# The library reads its config path at import time; write a POSIX stub
# before importing it.
constants.DB_CONFIG = "/config/db_stub.conf"
_DB_STUB = os.path.join(REPO, "config", "db_stub.conf")
with open(_DB_STUB, "w") as _f:
    _f.write("[SYSTEM]\ndb_type = POSTGRES\n\n[POSTGRES]\ndatabase = IMDB0001\n")

# Cost-only what-if: constants.COST_TYPE selects the cost signal.
constants.COST_TYPE_CURRENT_EXECUTION = constants.COST_TYPE_SUB_TREE_COST
constants.COST_TYPE_CURRENT_CREATION = constants.COST_TYPE_SUB_TREE_COST

import database.sql_helper_v2 as sql_helper  # noqa: E402
import bandits.bandit_c3ucb_v2 as bandit_module  # noqa: E402
import bandits.bandit_helper_v2 as bandit_helper  # noqa: E402
from bandits.query_v5 import Query  # noqa: E402
from bandits.oracle_v2 import OracleV7 as Oracle  # noqa: E402

import psycopg2  # noqa: E402
import sqlglot  # noqa: E402
from sqlglot import exp as sqlglot_exp  # noqa: E402
from sqlglot.optimizer.qualify import qualify as sqlglot_qualify  # noqa: E402

DEFAULT_STATE_DIR = _STATE_DIR_ENV
DEFAULT_CONFIG = {
    "model_dir": DEFAULT_STATE_DIR,
    "rounds": 25,              # what-if rounds per /recommend (round 0 = observation only)
    "train_trigger": 5,        # observed requests before background training fires
    "training_rounds": 10,     # total what-if rounds per background training run
    "training_workloads": 5,   # synthetic workloads sampled per training run
    "training_workload_size": 100,  # draws per synthetic workload
    "training_sample_seed": 0, # 0 = fresh seed per run (recorded)
    "time_budget_seconds": 60,  # wall-clock cap per request; 0 = unlimited
    "alpha": 1.0,              # C3UCB exploration hyper-parameter
    "lambda": 0.5,             # C3UCB regularizer
    "max_index_width": 2,      # max key columns per index arm; 0 = unrestricted
    "max_query_store": 2000,   # persisted query cap; 0 = uncapped
    "max_arm_store": 5000,     # persisted arm cap; 0 = uncapped
    "max_state_mb": 32.0,      # persisted state byte cap; 0 = uncapped
    "statement_timeout_ms": 120000,
    "probe_timeout_ms": 60000,
}

DEFAULT_STORAGE_MB = 1000  # used when --budget omits storage_mb
DEFAULT_MAX_INDEXES = 10   # used when --budget omits max_indexes

_STATE_FILE = "bandit_state.pkl"
_META_FILE = "model_meta.json"
_COUNTERS_FILE = "counters.json"
_TRAINING_FILE = "training.json"


class _QueryPlanError(Exception):
    """EXPLAIN failed for a query (unparseable / absent table / DDL error)."""


# ---------------------------------------------------------------------------
# PostgreSQL adapter: runtime monkey-patches of the database-access functions.
# The algorithm core (arm generation, C3UCB math, OracleV7, reward
# computation) is untouched and calls these through the sql_helper namespace.
# ---------------------------------------------------------------------------
_INTROSPECT_CACHE = {}
_HYP_NAME_MAP = {}   # hypopg internal index name -> arm index_name
_HYP_OID_MAP = {}    # arm index_name -> hypopg index oid

# Fixed-width types: catalog width in bytes. Variable-length types ('varchar'
# family, incl. text/bytea/json) get a probed max octet length.
_TYPE_WIDTH = {
    "smallint": 2, "integer": 4, "bigint": 8, "real": 4, "double precision": 8,
    "boolean": 1, "date": 4, "timestamp": 8, "timestamp without time zone": 8,
    "timestamp with time zone": 8, "time": 8, "time without time zone": 8,
    "interval": 12, "uuid": 16,
}
_VARIABLE_TYPES = {
    "varchar", "character varying", "character", "char", "text", "bytea",
    "json", "jsonb", "array", "USER-DEFINED", "xml",
}


def _conn_dsn(connection):
    return getattr(connection, "skill_dsn", None)


def _esc(text):
    """Escape % so psycopg2 does not treat it as a parameter placeholder."""
    return text.replace("%", "%%")


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


_QUALIFIED_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*)?$")
_QUOTED_IDENT_RE = re.compile(r'^"((?:[^"]|"")*)"(.*)$', re.DOTALL)


def _parse_index_key_part(part):
    """One key-column entry of a pg_indexes.indexdef -> lower-case column name,
    or None for an expression. Strips [COLLATE <name>] [opclass]
    [ASC|DESC] [NULLS FIRST|LAST]."""
    s = part.strip()
    if not s:
        return None
    m = _QUOTED_IDENT_RE.match(s)
    if m:
        name = m.group(1).replace('""', '"')
        rest = m.group(2).strip()
        tokens = rest.split() if rest else []
        quoted = True
    else:
        tokens = s.split()
        name = tokens[0]
        tokens = tokens[1:]
        quoted = False
    opclass_seen = False
    i = 0
    while i < len(tokens):
        tok = tokens[i].upper()
        if tok == "COLLATE":
            i += 2            # COLLATE <collation name>
            continue
        if tok == "NULLS":
            i += 2            # NULLS FIRST|LAST
            continue
        if tok in ("ASC", "DESC"):
            i += 1
            continue
        if not opclass_seen and _QUALIFIED_IDENT_RE.match(tokens[i].strip('"')):
            opclass_seen = True   # e.g. text_pattern_ops / pg_catalog.int4_ops
            i += 1
            continue
        return None
    if not name or (not quoted and not _IDENT_RE.match(name)):
        return None
    return name.lower()


def _parse_indexdef_key_columns(indexdef):
    """Key (btree) columns of a pg_indexes.indexdef; [] when the key cannot be
    read with certainty (expression or non-btree index). INCLUDE is not part
    of the key."""
    if not indexdef:
        return []
    m = re.search(r"\bON\b\s+\S+\s+USING\s+btree\s*\(([^)]*)\)", indexdef, re.I)
    if not m:
        m = re.search(r"\bON\b\s+\S+\s*\(([^)]*)\)", indexdef, re.I)
    if not m:
        return []
    cols = []
    for part in m.group(1).split(","):
        if not part.strip():
            continue
        name = _parse_index_key_part(part)
        if name is None:
            return []  # expression key — do not guess
        cols.append(name)
    return cols


def _resolve_table(meta, name):
    if name is None:
        return None
    if name in meta["tables"]:
        return name
    return meta["lower_table"].get(name.lower())


def _introspect(dsn, probe_timeout_ms=60000):
    """Immutable schema introspection per DSN (tables, columns, sizes, PKs)."""
    if dsn in _INTROSPECT_CACHE:
        return _INTROSPECT_CACHE[dsn]
    conn = psycopg2.connect(dsn, connect_timeout=10)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = %s", [int(probe_timeout_ms)])
    tables = {}
    cur.execute("""
        SELECT n.nspname, c.relname, c.reltuples::bigint, pg_total_relation_size(c.oid)
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind = 'r' AND n.nspname NOT IN ('pg_catalog', 'information_schema')
    """)
    for schema, tname, row_count, size_bytes in cur.fetchall():
        tables[tname] = {
            "schema": schema, "row_count": max(0, row_count or 0),
            "size_bytes": int(size_bytes or 0),
            "columns": [], "col_info": {}, "pk": [],
        }
    cur.execute("""
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_name, ordinal_position
    """)
    for c_schema, tname, cname, dtype in cur.fetchall():
        if tname not in tables:
            continue
        tables[tname]["columns"].append(cname)
        tables[tname]["col_info"][cname] = {"type": dtype, "avg_width": None, "max_width": None}
    # average column widths (pg_stats; missing stats fall back to type width)
    cur.execute("SELECT schemaname, tablename, attname, avg_width FROM pg_stats")
    for _s, tname, cname, avg_width in cur.fetchall():
        if tname in tables and cname in tables[tname]["col_info"]:
            tables[tname]["col_info"][cname]["avg_width"] = avg_width
    # primary keys
    cur.execute("""
        SELECT n.nspname, t.relname, a.attname, k.ord
        FROM pg_index i
        JOIN pg_class t ON t.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
        WHERE i.indisprimary
        ORDER BY k.ord
    """)
    for _s, tname, cname, _o in cur.fetchall():
        if tname in tables:
            tables[tname]["pk"].append(cname)
    # Column sizes: fixed types use the catalog width; variable types are
    # probed with max(octet_length(col)); on probe timeout the column is
    # treated as oversized.
    for tname, info in tables.items():
        text_cols = [c for c in info["columns"]
                     if info["col_info"][c]["type"].lower() in _VARIABLE_TYPES]
        for cname in info["columns"]:
            ci = info["col_info"][cname]
            if ci["type"].lower() not in _VARIABLE_TYPES:
                ci["max_width"] = _TYPE_WIDTH.get(ci["type"].lower(), 8)
                ci["avg_width"] = ci["avg_width"] or ci["max_width"]
        if text_cols:
            try:
                cur.execute("SET statement_timeout = %s", [int(probe_timeout_ms)])
                sel = ", ".join(f'max(octet_length("{c}"))' for c in text_cols)
                cur.execute(f'SELECT {sel} FROM "{info["schema"]}"."{tname}"')
                row = cur.fetchone()
                for cname, val in zip(text_cols, row):
                    tables[tname]["col_info"][cname]["max_width"] = val if val is not None else 0
                    tables[tname]["col_info"][cname]["avg_width"] = \
                        tables[tname]["col_info"][cname]["avg_width"] or 8
            except Exception:
                for cname in text_cols:
                    tables[tname]["col_info"][cname]["max_width"] = 999999
                    tables[tname]["col_info"][cname]["avg_width"] = \
                        tables[tname]["col_info"][cname]["avg_width"] or 8
    cur.execute("SELECT pg_database_size(current_database())")
    db_size = float(cur.fetchone()[0])
    conn.close()

    all_columns = {t: list(info["columns"]) for t, info in tables.items()}
    meta = {
        "tables": tables,
        "lower_table": {t.lower(): t for t in tables},
        "lower_col": {t.lower(): {c.lower(): c for c in info["columns"]}
                      for t, info in tables.items()},
        "all_columns": all_columns,
        "n_columns": sum(len(v) for v in all_columns.values()),
        "db_size": db_size,
        "scan_structure": {t: [] for t in tables},
        "table_objects": None,
    }
    _INTROSPECT_CACHE[dsn] = meta
    return meta


def _meta_for(connection):
    dsn = _conn_dsn(connection)
    if dsn is None:
        raise RuntimeError("connection was not created by the wrapper")
    return _introspect(dsn)


class _ConnProxy:
    """psycopg2 connections reject attribute assignment; carry the DSN tag on a
    proxy that exposes the pyodbc-style surface (cursor/commit/close)."""

    def __init__(self, conn, dsn):
        self._conn = conn
        self.skill_dsn = dsn

    def cursor(self, *a, **kw):
        return self._conn.cursor(*a, **kw)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()


def _pg_connect(dsn, statement_timeout_ms=120000):
    conn = psycopg2.connect(dsn, connect_timeout=10)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = %s", [int(statement_timeout_ms)])
    cur.close()
    return _ConnProxy(conn, dsn)


def _build_table_objects(meta):
    """Table/Column objects (the shapes get_tables/get_columns return)."""
    if meta["table_objects"] is not None:
        return meta["table_objects"]
    objs = {}
    for tname, info in meta["tables"].items():
        table = sql_helper.Table(tname, info["row_count"], list(info["pk"]))
        cols = {}
        for cname in info["columns"]:
            ci = info["col_info"][cname]
            # Map variable-length PostgreSQL types to the 'varchar' branch
            # (variable key overhead).
            ctype = "varchar" if ci["type"].lower() in _VARIABLE_TYPES else ci["type"]
            col = sql_helper.Column(tname, cname, ctype)
            col.set_column_size(int(ci["avg_width"] or 8))
            col.set_max_column_size(ci["max_width"] if ci["max_width"] is not None else 0)
            cols[cname] = col
        table.set_columns(cols)
        objs[tname] = table
    meta["table_objects"] = objs
    return objs


def _walk_plan_nodes(node):
    yield node
    for child in node.get("Plans", []) or []:
        yield from _walk_plan_nodes(child)


def _explain_json(cursor, query):
    cursor.execute("EXPLAIN (FORMAT JSON) " + _esc(query))
    rows = cursor.fetchone()[0]
    return rows[0]["Plan"] if rows else {}


# --- patched catalog / plan functions (sql_helper module attributes) ------

def _patched_get_tables(connection):
    meta = _meta_for(connection)
    objs = _build_table_objects(meta)
    sql_helper.tables_global = objs
    sql_helper.pk_columns_dict = {t: list(i["pk"]) for t, i in meta["tables"].items()}
    return objs


def _patched_get_table_row_count(connection, schema_name, tbl_name):
    meta = _meta_for(connection)
    t = _resolve_table(meta, tbl_name)
    if t is None:
        return 0
    return meta["tables"][t]["row_count"]


def _patched_get_primary_key(connection, schema_name, table_name):
    meta = _meta_for(connection)
    t = _resolve_table(meta, table_name)
    if t is None:
        return []
    return list(meta["tables"][t]["pk"])


def _patched_get_columns(connection, table_name):
    meta = _meta_for(connection)
    objs = _build_table_objects(meta)
    t = _resolve_table(meta, table_name)
    if t is None:
        return {}
    return objs[t].get_columns()


def _patched_get_all_columns(connection):
    meta = _meta_for(connection)
    return meta["all_columns"], meta["n_columns"]


def _patched_get_database_size(connection):
    meta = _meta_for(connection)
    return meta["db_size"]


def _patched_get_table_scan_times_structure():
    meta = _INTROSPECT_CACHE.get(_CURRENT_DSN[0]) if _CURRENT_DSN[0] else None
    if meta is None:
        # not bound yet (should not happen on the request path)
        return {t: [] for t in sql_helper.tables_global or {}}
    return {t: [] for t in meta["tables"]}


def _patched_hyp_create_index_v1(connection, schema_name, tbl_name, col_names,
                                 idx_name, include_cols=()):
    """Create a hypothetical index with hypopg_create_index."""
    meta = _meta_for(connection)
    t = _resolve_table(meta, tbl_name)
    if t is None:
        return 0
    schema = meta["tables"][t]["schema"]
    cols = ", ".join(f'"{c}"' for c in col_names)
    ddl = f'CREATE INDEX "{idx_name}" ON "{schema}"."{t}" ({cols})'
    if include_cols:
        incl = ", ".join(f'"{c}"' for c in include_cols)
        ddl += f" INCLUDE ({incl})"
    cur = connection.cursor()
    cur.execute("SELECT hypopg_create_index(%s)", [ddl])
    raw = cur.fetchone()[0]  # '(oid,<oid>btree_...)'
    oid_part, hyp_name = raw.strip("()").split(",", 1)
    _HYP_NAME_MAP[hyp_name.strip()] = idx_name
    _HYP_OID_MAP[idx_name] = int(oid_part.strip())
    return 0


def _patched_hyp_bulk_create_indexes(connection, schema_name, bandit_arm_list):
    cost = {}
    for index_name, bandit_arm in bandit_arm_list.items():
        cost[index_name] = _patched_hyp_create_index_v1(
            connection, schema_name, bandit_arm.table_name, bandit_arm.index_cols,
            bandit_arm.index_name, bandit_arm.include_cols)
    return cost


def _patched_hyp_enable_index(connection):
    # HypoPG indexes are session-local and used by EXPLAIN automatically.
    return None


def _patched_hyp_execute_query(connection, query):
    """Cost-only EXPLAIN under the session's hypothetical indexes.

    Returns (sub_tree_cost, non_clustered_index_usage, clustered_index_usage),
    where usage entries are (name, elapsed, cpu, po_subtree_cost, rows_read,
    rows_out) tuples. Table-access entries are emitted for every scan node;
    index-use entries only for the session's hypothetical indexes.
    """
    meta = _meta_for(connection)
    cur = connection.cursor()
    plan = _explain_json(cur, query)
    top_cost = float(plan.get("Total Cost") or 0.0)
    non_clustered, clustered = [], []
    for node in _walk_plan_nodes(plan):
        total_cost = float(node.get("Total Cost") or 0.0)
        rows = float(node.get("Plan Rows") or 0.0)
        rel = node.get("Relation Name")
        idx = node.get("Index Name")
        if rel:
            t = _resolve_table(meta, rel)
            if t is not None:
                clustered.append((t, total_cost, total_cost, total_cost, rows, rows))
        if idx:
            arm_name = _HYP_NAME_MAP.get(idx)
            if arm_name is not None:
                non_clustered.append((arm_name, total_cost, total_cost, total_cost, rows, rows))
    return top_cost, non_clustered, clustered


def _patched_bulk_drop_index(connection, schema_name, bandit_arm_list):
    cur = connection.cursor()
    for index_name, _arm in bandit_arm_list.items():
        oid = _HYP_OID_MAP.pop(index_name, None)
        if oid is not None:
            try:
                cur.execute("SELECT hypopg_drop_index(%s)", [oid])
                cur.fetchone()
            except Exception:
                pass
        for hyp_name in [k for k, v in _HYP_NAME_MAP.items() if v == index_name]:
            _HYP_NAME_MAP.pop(hyp_name, None)


def _patched_get_query_plan(connection, query):
    """PG replacement of the SHOWPLAN_XML reader (used by get_selectivity_v3)."""
    meta = _meta_for(connection)
    cur = connection.cursor()
    try:
        cur.execute("EXPLAIN (FORMAT JSON) " + _esc(query))
        rows = cur.fetchone()[0]
    except Exception as e:
        raise _QueryPlanError(str(e))
    return rows[0] if rows else {}


def _patched_get_selectivity_v3(connection, query, predicates):
    """Selectivity per predicate table from a cost-only EXPLAIN (plan rows)."""
    meta = _meta_for(connection)
    cur = connection.cursor()
    try:
        plan = _explain_json(cur, query)
    except Exception as e:
        raise _QueryPlanError(str(e))
    if not plan:
        return 1
    read_rows = {t: 1000000000.0 for t in predicates}
    for node in _walk_plan_nodes(plan):
        rel = node.get("Relation Name")
        if not rel:
            continue
        t = _resolve_table(meta, rel)
        if t in read_rows:
            rows = float(node.get("Plan Rows") or 0.0)
            read_rows[t] = min(read_rows[t], rows)
    selectivity = {}
    for t in predicates:
        row_count = meta["tables"].get(t, {}).get("row_count", 0)
        selectivity[t] = read_rows[t] / row_count if row_count else 1.0
    return selectivity


def _patched_restart_sql_server(*_a, **_kw):
    # No-op teardown hook.
    return None


_CURRENT_DSN = [None]

sql_helper.get_tables = _patched_get_tables
sql_helper.get_table_row_count = _patched_get_table_row_count
sql_helper.get_primary_key = _patched_get_primary_key
sql_helper.get_columns = _patched_get_columns
sql_helper.get_all_columns = _patched_get_all_columns
sql_helper.get_database_size = _patched_get_database_size
sql_helper.get_table_scan_times_structure = _patched_get_table_scan_times_structure
sql_helper.get_selectivity_v3 = _patched_get_selectivity_v3
sql_helper.get_query_plan = _patched_get_query_plan
sql_helper.hyp_create_index_v1 = _patched_hyp_create_index_v1
sql_helper.hyp_bulk_create_indexes = _patched_hyp_bulk_create_indexes
sql_helper.hyp_enable_index = _patched_hyp_enable_index
sql_helper.hyp_execute_query = _patched_hyp_execute_query
sql_helper.bulk_drop_index = _patched_bulk_drop_index
sql_helper.restart_sql_server = _patched_restart_sql_server


def _bind_dsn(dsn):
    """Bind the sql_helper module globals to the live dataset (dynamic
    replacement of the hard-coded TABLE_SCAN_TIMES per-database keys)."""
    _CURRENT_DSN[0] = dsn
    meta = _introspect(dsn)
    structure = {t: [] for t in meta["tables"]}
    sql_helper.table_scan_times_hyp = structure
    sql_helper.table_scan_times = {t: [] for t in meta["tables"]}
    sql_helper.tables_global = None
    return meta


# ---------------------------------------------------------------------------
# Workload parsing: predicates and payload are derived from raw SQL with
# sqlglot (WHERE/JOIN-ON columns become predicates, other referenced columns
# payload). JSON entries ({query_string, predicates, payload}) pass through.
# ---------------------------------------------------------------------------
_COMMENT_RE = re.compile(r"--[^\n]*")
_DIRECTIVE_RE = re.compile(r"^:[A-Za-z]$", re.M)


def _clean_stmt(text):
    text = _COMMENT_RE.sub(" ", text or "")
    text = _DIRECTIVE_RE.sub(" ", text)
    return text.strip().rstrip(";").strip()


def _split_statements(content):
    content = content.strip()
    if ";" not in content:
        return [content] if content else []
    return [s for s in (p.strip() for p in content.split(";")) if s.strip()]


def _is_read_query(text):
    head = (text or "").lstrip().lower()
    return head.startswith("select") or head.startswith("with")


def _resolve_col(meta, table, col):
    t = _resolve_table(meta, table)
    if t is None:
        return None
    c = meta["lower_col"][t.lower()].get((col or "").lower())
    return (t, c) if c else None


def _derive_predicates_payload(sql, meta):
    """(predicates, payload) dicts for a SQL statement."""
    stmt = sqlglot.parse_one(sql, read="postgres")
    schema = {t: {c: "int" for c in cols} for t, cols in meta["all_columns"].items()}
    try:
        stmt = sqlglot_qualify(stmt, schema=schema, dialect="postgres",
                               validate_qualify_columns=False)
    except Exception:
        pass  # best-effort qualification; alias resolution below still applies

    alias_map = {}
    real_tables = set()
    for t in stmt.find_all(sqlglot_exp.Table):
        actual = _resolve_table(meta, t.name)
        if actual is not None:
            real_tables.add(actual)
            if t.alias:
                alias_map[t.alias.lower()] = actual
            alias_map[t.name.lower()] = actual
    single_table = list(real_tables)[0] if len(real_tables) == 1 else None

    def _owner(column):
        tbl = alias_map.get((column.table or "").lower())
        if tbl is None and single_table:
            tbl = single_table
        if tbl is None:
            return None
        c = meta["lower_col"][tbl.lower()].get((column.name or "").lower())
        return (tbl, c) if c else None

    pred_nodes = []
    where = stmt.find(sqlglot_exp.Where)
    if where is not None:
        pred_nodes = list(where.find_all(sqlglot_exp.Column))
    for join in stmt.find_all(sqlglot_exp.Join):
        on = join.args.get("on")
        if on is not None:
            pred_nodes.extend(on.find_all(sqlglot_exp.Column))
    pred_set = set()
    for c in pred_nodes:
        owner = _owner(c)
        if owner:
            pred_set.add(owner)

    all_set = set()
    for c in stmt.find_all(sqlglot_exp.Column):
        owner = _owner(c)
        if owner:
            all_set.add(owner)

    predicates, payload = {}, {}
    for tbl, col in sorted(pred_set):
        predicates.setdefault(tbl, {})[col] = "e"
    for tbl, col in sorted(all_set - pred_set):
        payload.setdefault(tbl, []).append(col)
    return predicates, payload


def _normalize_repo_entry(meta, predicates, payload):
    """Case-insensitive resolution of a repo-format workload entry."""
    out_p, out_pl = {}, {}
    for tbl, cols in (predicates or {}).items():
        t = _resolve_table(meta, tbl)
        if t is None:
            continue
        for col, op in cols.items():
            c = meta["lower_col"][t.lower()].get((col or "").lower())
            if c:
                out_p.setdefault(t, {})[c] = op
    for tbl, cols in (payload or {}).items():
        t = _resolve_table(meta, tbl)
        if t is None:
            continue
        for col in cols or []:
            c = meta["lower_col"][t.lower()].get((col or "").lower())
            if c:
                out_pl.setdefault(t, []).append(c)
    return out_p, out_pl


def _parse_workload(workload_field, meta):
    """Normalize the workload input into entries: [{sql, weight, predicates,
    payload}] — predicates/payload present only for repo-format entries."""
    entries = []
    skipped_dml = [0]

    def _add(sql, weight=1, predicates=None, payload=None):
        sql = _clean_stmt(sql)
        if not sql:
            return
        if not _is_read_query(sql):
            skipped_dml[0] += 1  # write cost is estimated from the catalog, not measured
            return
        if predicates is None and payload is None:
            # Derive predicates + payload with sqlglot; failure degrades only
            # this query.
            try:
                predicates, payload = _derive_predicates_payload(sql, meta)
            except Exception:
                entries.append({"sql": sql, "weight": weight,
                                "predicates": {}, "payload": {}, "unparsed": True})
                return
        entries.append({"sql": sql, "weight": weight,
                        "predicates": predicates, "payload": payload})

    def _add_entry(e):
        if isinstance(e, str):
            _add(e)
        elif isinstance(e, dict):
            if "query_string" in e:
                p, pl = _normalize_repo_entry(meta, e.get("predicates"), e.get("payload"))
                _add(e["query_string"], e.get("weight", e.get("frequency", 1)), p, pl)
            elif "sql" in e or "query" in e:
                _add(e.get("sql") or e.get("query"), e.get("weight", e.get("frequency", 1)))
        else:
            _add(str(e))

    if isinstance(workload_field, list):
        for e in workload_field:
            _add_entry(e)
    elif isinstance(workload_field, str):
        text = workload_field.strip()
        as_json = None
        if text.startswith("[") or text.startswith("{"):
            try:
                as_json = json.loads(text)
            except Exception:
                as_json = None
        if as_json is not None:
            sub_entries, sub_dml = _parse_workload(as_json, meta)
            entries.extend(sub_entries)
            skipped_dml[0] += sub_dml
        elif os.path.exists(text):
            if os.path.isdir(text):
                files = []
                for root, _d, fs in os.walk(text):
                    for f in fs:
                        if f.lower().endswith((".sql", ".json", ".jsonl")):
                            files.append(os.path.join(root, f))
                for fp in sorted(files):
                    _load_file(fp, _add)
            else:
                _load_file(text, _add)
        else:
            for stmt in _split_statements(text):
                _add(stmt)
    return entries, skipped_dml[0]


def _load_file(fp, add_fn):
    try:
        with open(fp, "r") as fh:
            content = fh.read()
    except Exception:
        return
    if fp.lower().endswith((".json", ".jsonl")):
        try:
            data = json.loads(content)
        except Exception:
            try:
                data = [json.loads(line) for line in content.splitlines() if line.strip()]
            except Exception:
                return
        if isinstance(data, dict):
            data = [data]
        for e in data if isinstance(data, list) else []:
            if isinstance(e, str):
                add_fn(e)
            elif isinstance(e, dict) and "query_string" in e:
                add_fn(e["query_string"], e.get("weight", e.get("frequency", 1)),
                       e.get("predicates"), e.get("payload"))
            elif isinstance(e, dict) and ("sql" in e or "query" in e):
                add_fn(e.get("sql") or e.get("query"), e.get("weight", 1))
    else:
        for stmt in _split_statements(content):
            add_fn(stmt)


# ---------------------------------------------------------------------------
# The skill server
# ---------------------------------------------------------------------------
def _atomic_write(path, data, binary=False):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb" if binary else "w") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_pickle(path, obj):
    data = obj if isinstance(obj, (bytes, bytearray)) else pickle.dumps(
        obj, protocol=pickle.HIGHEST_PROTOCOL)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class DBABanditsServer:
    def __init__(self, state_dir=DEFAULT_STATE_DIR):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._shutdown = False
        self._training_pid = None
        self._counters = {
            "requests_served": 0,
            "rounds_total": 0,
            "queries_analyzed_total": 0,
            "observations_since_training": 0,
            "training_runs": 0,
            "recommend_only_requests": 0,
        }
        self._last_config = dict(DEFAULT_CONFIG)
        self._bandit = None
        self._bandit_epoch = 0
        self._live_context_size = None
        self._live_n_columns = None
        # learning state (loaded lazily / rehydrated from disk)
        self._bandit_state = None       # {"v","b","hyper_alpha","context_size","n_columns"}
        self._query_store = {}          # query id -> Query
        self._state_mtime = 0.0
        self._last_state_bytes = 0
        self._state_over_budget = False
        self._last_evicted_queries = 0
        self._last_evicted_arms = 0
        self._evicted_query_ids = set()
        self._evicted_order = deque()
        # The revision/matrices this process loaded from disk: a concurrent
        # writer (the background training subprocess) may save a newer state
        # while a request is in flight; bandit updates are additive, so on a
        # conflicting save the writer re-applies its DELTA on top of the
        # newer on-disk matrices instead of clobbering them.
        self._state_base = None
        self._base_rev = 0
        self._load_from_disk(initial=True)

    # -- persistence --------------------------------------------------------

    def _state_paths(self):
        return (os.path.join(self.state_dir, _STATE_FILE),
                os.path.join(self.state_dir, _META_FILE),
                os.path.join(self.state_dir, _COUNTERS_FILE))

    def _remember_evicted(self, ids):
        cap = int(self._last_config.get("max_query_store", 2000) or 2000)
        for qid in ids:
            if qid not in self._evicted_query_ids:
                self._evicted_query_ids.add(qid)
                self._evicted_order.append(qid)
        limit = max(16, cap * 4)
        while len(self._evicted_order) > limit:
            old = self._evicted_order.popleft()
            self._evicted_query_ids.discard(old)

    def _apply_state_caps(self, cfg, query_cap=None, arm_cap=None):
        """Bound the persisted stores; mutates in-memory state so a later
        merge cannot resurrect evicted entries."""
        qcap = int(query_cap if query_cap is not None
                   else (cfg.get("max_query_store", 2000) or 0))
        acap = int(arm_cap if arm_cap is not None
                   else (cfg.get("max_arm_store", 5000) or 0))
        store = self._query_store
        evicted_q = 0
        if qcap > 0 and len(store) > qcap:
            ranked = sorted(
                store.items(),
                key=lambda kv: (getattr(kv[1], "last_request_at", 0) or 0,
                                getattr(kv[1], "frequency", 0) or 0,
                                kv[0]),
                reverse=True)[:qcap]
            kept = {qid for qid, _ in ranked}
            evicted = [qid for qid in store if qid not in kept]
            self._query_store = dict(ranked)
            store = self._query_store
            evicted_q = len(evicted)
            self._remember_evicted(evicted)
        live = set(store)
        arms = bandit_helper.bandit_arm_store
        survivors = {}
        for key, arm in arms.items():
            if getattr(arm, "arm_value", None):
                arm.arm_value = {qid: v for qid, v in arm.arm_value.items()
                                 if qid in live}
            for attr in ("query_ids", "query_ids_backup"):
                s = getattr(arm, attr, None)
                if isinstance(s, (set, frozenset)):
                    setattr(arm, attr, set(s) & live)
            if (getattr(arm, "query_id", None) is not None
                    and arm.query_id not in live):
                arm.query_id = None
            if (not getattr(arm, "arm_value", None)
                    and not getattr(arm, "query_ids", None)
                    and getattr(arm, "query_id", None) is None):
                continue
            survivors[key] = arm
        if acap > 0 and len(survivors) > acap:
            def score(item):
                a = item[1]
                return (float(sum((getattr(a, "arm_value", None) or {}).values())),
                        len(getattr(a, "query_ids", ()) or ()),
                        str(getattr(a, "index_name", "")),
                        str(getattr(a, "table_name", "")),
                        tuple(str(c) for c in (getattr(a, "index_cols", ()) or ())),
                        item[0])
            survivors = dict(sorted(survivors.items(), key=score,
                                    reverse=True)[:acap])
        evicted_a = len(arms) - len(survivors)
        arms.clear()
        arms.update(survivors)
        return evicted_q, evicted_a

    def _load_from_disk(self, initial=False):
        pkl, meta_path, cnt_path = self._state_paths()
        try:
            mtime = os.path.getmtime(pkl)
        except OSError:
            return
        if mtime <= self._state_mtime:
            return
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            self._bandit_state = {
                "v": numpy.array(data["v"], dtype=float),
                "b": numpy.array(data["b"], dtype=float),
                "hyper_alpha": float(data["hyper_alpha"]),
                "context_size": int(data["context_size"]),
                "n_columns": int(data["n_columns"]),
            }
            self._query_store = data.get("query_store") or {}
            self._counters.update(data.get("counters") or {})
            self._last_config = dict(DEFAULT_CONFIG)
            self._last_config.update(data.get("last_config") or {})
            # the module-global arm store IS the learned arm state
            bandit_helper.bandit_arm_store.clear()
            bandit_helper.bandit_arm_store.update(data.get("arm_store") or {})
            self._last_evicted_queries, self._last_evicted_arms = \
                self._apply_state_caps(self._last_config)
            self._state_mtime = mtime
            self._state_base = {
                "v": numpy.array(self._bandit_state["v"], dtype=float),
                "b": numpy.array(self._bandit_state["b"], dtype=float),
                "alpha": float(self._bandit_state["hyper_alpha"]),
            }
            self._base_rev = int(data.get("state_rev") or 0)
            self._bandit_epoch += 1
            self._bandit = None
            if not initial:
                self._counters["observations_since_training"] = 0
        except Exception:
            # a corrupt or foreign pickle must never take the server down
            self._bandit_state = None

    def _read_disk_state(self):
        pkl = self._state_paths()[0]
        try:
            with open(pkl, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _persist_state(self):
        bandit = getattr(self, "_bandit", None)
        if bandit is not None:
            v = numpy.array(bandit.v, dtype=float)
            b = numpy.array(bandit.b, dtype=float)
            alpha = float(bandit.hyper_alpha)
        else:
            v = numpy.array((self._bandit_state or {}).get("v"), dtype=float)
            b = numpy.array((self._bandit_state or {}).get("b"), dtype=float)
            alpha = float((self._bandit_state or {}).get("hyper_alpha") or 1.0)

        # conflict resolution against a concurrent writer (training subprocess
        # and/or a previous server process): apply OUR delta (current minus
        # the matrices we loaded) on top of the newer on-disk matrices.
        disk = self._read_disk_state()
        disk_rev = int((disk or {}).get("state_rev") or 0)
        conflict = (disk is not None
                    and disk_rev > self._base_rev
                    and self._state_base is not None
                    and int(disk.get("context_size") or -1) == (self._context_size() or -2))
        if conflict:
            try:
                dv = v - self._state_base["v"]
                db = b - self._state_base["b"]
                v = numpy.array(disk["v"], dtype=float) + dv
                b = numpy.array(disk["b"], dtype=float) + db
                base_alpha = self._state_base["alpha"] or 1.0
                alpha = float(disk["hyper_alpha"]) * (alpha / base_alpha)
                merged_queries = dict(self._query_store)
                merged_queries.update(disk.get("query_store") or {})
                merged_arms = dict(bandit_helper.bandit_arm_store)
                merged_arms.update(disk.get("arm_store") or {})
                counters = dict(self._counters)
                disk_counters = disk.get("counters") or {}
                for k in ("requests_served", "rounds_total", "queries_analyzed_total",
                          "recommend_only_requests", "training_runs"):
                    counters[k] = max(counters.get(k, 0), disk_counters.get(k, 0))
                self._query_store = merged_queries
                self._counters = counters
                bandit_helper.bandit_arm_store.clear()
                bandit_helper.bandit_arm_store.update(merged_arms)
                if bandit is not None:
                    bandit.v = v
                    bandit.b = b
                    bandit.hyper_alpha = alpha
            except Exception:
                pass  # shapes incompatible — fall through to a plain save

        self._last_evicted_queries, self._last_evicted_arms = \
            self._apply_state_caps(self._last_config)
        new_rev = max(disk_rev, self._base_rev) + 1
        payload = {
            "v": v,
            "b": b,
            "hyper_alpha": alpha,
            "context_size": self._context_size(),
            "n_columns": self._n_columns(),
            "arm_store": dict(bandit_helper.bandit_arm_store),
            "query_store": self._query_store,
            "counters": dict(self._counters),
            "last_config": self._last_config,
            "state_rev": new_rev,
        }
        pkl, meta_path, cnt_path = self._state_paths()
        cfg = self._last_config or DEFAULT_CONFIG
        try:
            limit_mb = float(cfg.get("max_state_mb", 32.0) or 0)
        except (TypeError, ValueError):
            limit_mb = float(DEFAULT_CONFIG["max_state_mb"])
        limit = int(limit_mb * 1024 * 1024) if limit_mb > 0 else 0
        qcap = int(cfg.get("max_query_store", 2000) or 0)
        acap = int(cfg.get("max_arm_store", 5000) or 0)
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        for _ in range(6):
            if not limit or len(blob) <= limit:
                break
            qcap = max(1, (qcap or len(self._query_store)) // 2)
            acap = max(1, (acap or len(bandit_helper.bandit_arm_store)) // 2)
            eq, ea = self._apply_state_caps(cfg, qcap, acap)
            self._last_evicted_queries += eq
            self._last_evicted_arms += ea
            payload["query_store"] = self._query_store
            payload["arm_store"] = dict(bandit_helper.bandit_arm_store)
            blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self._last_state_bytes = len(blob)
        self._state_over_budget = bool(limit and len(blob) > limit)
        _atomic_pickle(pkl, blob)
        self._state_mtime = os.path.getmtime(pkl)
        self._state_base = {"v": numpy.array(v, dtype=float),
                            "b": numpy.array(b, dtype=float),
                            "alpha": alpha}
        self._base_rev = new_rev
        meta = {
            "context_size": self._context_size(),
            "n_columns": self._n_columns(),
            "saved_at": time.time(),
            "state_dir": self.state_dir,
        }
        _atomic_write(meta_path, json.dumps(meta, indent=2))
        _atomic_write(cnt_path, json.dumps(dict(self._counters), indent=2))

    def _context_size(self):
        if self._bandit_state is not None:
            return self._bandit_state["context_size"]
        return getattr(self, "_live_context_size", None)

    def _n_columns(self):
        if self._bandit_state is not None:
            return self._bandit_state["n_columns"]
        return getattr(self, "_live_n_columns", None)

    def _maybe_reload_from_disk(self):
        self._load_from_disk()

    # -- training subprocess --------------------------------------------------

    def _training_alive(self):
        pid = self._training_pid
        if pid is None:
            return False
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().decode(errors="replace")
            return "wrapper.py" in cmdline and "--train" in cmdline
        except OSError:
            self._training_pid = None
            return False

    def _maybe_spawn_training(self, eval_dsn, cfg):
        if self._training_alive() or self._shutdown:
            return False
        if self._counters["observations_since_training"] < int(cfg["train_trigger"]):
            return False
        cmd = [sys.executable, os.path.join(REPO, "wrapper.py"),
               "--train", "--model-dir", self.state_dir, "--eval-dsn", eval_dsn]
        try:
            proc = subprocess.Popen(cmd, start_new_session=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._training_pid = proc.pid
            self._counters["training_runs"] += 1
            _atomic_write(os.path.join(self.state_dir, _TRAINING_FILE),
                          json.dumps({"pid": proc.pid, "status": "running",
                                      "started_at": time.time()}))
            print(f"[train] spawned background training pid={proc.pid} "
                  f"rounds={cfg['training_rounds']} "
                  f"workloads={cfg['training_workloads']}", flush=True)
            return True
        except Exception as e:
            print(f"[train] spawn skipped: {type(e).__name__}", flush=True)
            return False

    def _sample_training_pools(self, cfg):
        """Synthetic workloads: draws from the store with replacement, so the
        composition and multiplicity differ from the stored workload."""
        qs = list(self._query_store.values())
        if not qs:
            return [], 0, []
        n_pools = max(1, int(cfg.get("training_workloads", 5) or 1))
        size = int(cfg.get("training_workload_size", 100) or 0)
        seed = int(cfg.get("training_sample_seed", 0) or 0)
        if seed == 0:
            seed = (int(time.time()) ^ os.getpid()) & 0x7fffffff
        rng = random.Random(seed)
        weights = [max(1, int(getattr(q, "frequency", 1) or 1)) for q in qs]
        pools, sizes = [], []
        for _ in range(n_pools):
            if size <= 0 or size >= len(qs):
                draws = list(qs)
            else:
                draws = rng.choices(qs, weights=weights, k=size)
            counts = {}
            pool = []
            for q in draws:
                try:
                    clone = copy.deepcopy(q)
                except Exception:
                    clone = copy.copy(q)
                    clone.predicates = copy.deepcopy(q.predicates)
                    clone.payload = copy.deepcopy(q.payload)
                    clone.table_scan_times = copy.deepcopy(q.table_scan_times)
                    clone.index_scan_times = copy.deepcopy(q.index_scan_times)
                    clone.table_scan_times_hyp = copy.deepcopy(q.table_scan_times_hyp)
                    clone.index_scan_times_hyp = copy.deepcopy(q.index_scan_times_hyp)
                counts[q.id] = counts.get(q.id, 0) + 1
                clone.frequency = counts[q.id]
                clone.last_request_at = getattr(q, "last_request_at", 0)
                pool.append(clone)
            pools.append(pool)
            sizes.append(len(pool))
        return pools, seed, sizes

    def train_once(self, eval_dsn, rounds=None):
        """Background-training entry point (`wrapper.py --train`)."""
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(self._last_config or {})
        conn = _pg_connect(eval_dsn, cfg["statement_timeout_ms"])
        try:
            cur = conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
        except Exception:
            print("[train] hypopg unavailable on eval dsn; training aborted", flush=True)
            return
        meta = _bind_dsn(eval_dsn)
        self._live_context_size = meta["n_columns"] + constants.STATIC_CONTEXT_SIZE
        self._live_n_columns = meta["n_columns"]
        self._load_from_disk(initial=True)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(self._last_config or {})
        if rounds:
            cfg["training_rounds"] = int(rounds)
        if not self._query_store:
            print("[train] no persisted queries; nothing to train on", flush=True)
            return
        pools, seed, sizes = self._sample_training_pools(cfg)
        if not pools:
            print("[train] could not sample training workloads", flush=True)
            return
        total_rounds = max(1, int(cfg.get("training_rounds", 10) or 1))
        n_pools = min(len(pools), max(1, total_rounds // 2))
        pools, sizes = pools[:n_pools], sizes[:n_pools]
        base = max(1, total_rounds // n_pools)
        extra = max(0, total_rounds - base * n_pools)
        self._ensure_bandit(cfg, DEFAULT_STORAGE_MB)
        _tb = float(cfg["time_budget_seconds"])
        deadline = time.time() + (2.0 * _tb if _tb > 0 else 1e9)
        executed = 0
        with self._lock:
            for i, pool in enumerate(pools):
                if time.time() > deadline:
                    break
                result = self._run_rounds(
                    conn, meta, cfg, pool, deadline,
                    base + (1 if i < extra else 0), pool=pool)
                executed += result["rounds_executed"]
                if result["time_budget_hit"]:
                    break
            self._counters["rounds_total"] += executed
            self._counters["observations_since_training"] = 0
            self._counters["last_training_sample_seed"] = seed
            self._counters["last_training_pool_sizes"] = sizes
            self._persist_state()
        print(f"[train] pools={n_pools} sizes={sizes} seed={seed} "
              f"rounds={executed}", flush=True)

    # -- recommendation -------------------------------------------------------

    def recommend(self, req):
        t0 = time.time()
        with self._lock:
            return self._recommend_locked(req, t0)

    def _recommend_locked(self, req, t0):
        dsn = req.get("dsn") or ""
        eval_dsn = req.get("eval_dsn") or dsn
        recommend_only = bool(req.get("recommend_only"))
        apply_flag = bool(req.get("apply"))
        budget = self._parse_budget(req.get("budget"))
        cfg = self._parse_config(req.get("config"))
        notes = []

        if not dsn:
            return self._degrade("missing --dsn", cfg, 0, notes,
                                 recommend_only=recommend_only, budget=budget)

        # a --config model_dir change reloads the learning state
        if cfg["model_dir"] and os.path.abspath(cfg["model_dir"]) != os.path.abspath(self.state_dir):
            with_lock_note = self._persist_state()
            self.state_dir = cfg["model_dir"]
            os.makedirs(self.state_dir, exist_ok=True)
            self._bandit_state = None
            self._bandit = None
            self._query_store = {}
            self._state_mtime = 0.0
            self._bandit_epoch += 1
            bandit_helper.bandit_arm_store.clear()
            self._load_from_disk(initial=True)
            notes.append(f"learning state reloaded from {self.state_dir}")

        _tb = float(cfg["time_budget_seconds"])
        deadline = time.time() + (_tb if _tb > 0 else 1e9)
        eval_conn = None
        try:
            try:
                eval_conn = _pg_connect(eval_dsn, cfg["statement_timeout_ms"])
                cur = eval_conn.cursor()
                cur.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
            except Exception as e:
                return self._degrade(f"cannot reach eval database: {type(e).__name__}",
                                     cfg, 0, notes, recommend_only=recommend_only,
                                     budget=budget)

            meta = _bind_dsn(eval_dsn)
            self._live_context_size = meta["n_columns"] + constants.STATIC_CONTEXT_SIZE
            self._live_n_columns = meta["n_columns"]

            # learning state: reload (training subprocess) + space gate
            self._maybe_reload_from_disk()
            space_ok = (self._bandit_state is None
                        or (self._bandit_state["context_size"] == self._live_context_size
                            and self._bandit_state["n_columns"] == self._live_n_columns))
            if not space_ok:
                self._bandit_state = None
                self._bandit = None
                self._bandit_epoch += 1
                self._query_store = {}
                bandit_helper.bandit_arm_store.clear()
                notes.append("learned state discarded: schema width changed (fresh bootstrap)")

            # workload -> query objects
            entries, skipped_dml = _parse_workload(req.get("workload"), meta)
            if skipped_dml:
                notes.append(f"{skipped_dml} DML statements excluded from read analysis "
                             "(write overhead is estimated from the catalog)")
            queries, skipped = self._build_queries(eval_conn, entries, meta)
            for why in skipped:
                notes.append(f"query skipped: {why}")
            if not queries:
                note = "; ".join(notes) if notes else "workload contained no analyzable read queries"
                return self._degrade(note, cfg, 0, notes,
                                     recommend_only=recommend_only, budget=budget)

            # register in the persistent query store
            store_size_before = len(self._query_store)
            additions = [q for q in queries if q.id not in self._query_store]
            for q in additions:
                self._query_store[q.id] = q

            max_memory = float(budget.get("storage_mb") or DEFAULT_STORAGE_MB)
            self._ensure_bandit(cfg, max_memory)

            # Workload-change trigger (history is discarded when most queries
            # are new). Re-seen evicted queries do not count as new.
            if not recommend_only and additions and self._bandit is not None:
                new_count = sum(1 for q in additions
                                if q.id not in self._evicted_query_ids)
                if store_size_before > 0 and new_count > 0:
                    try:
                        self._bandit.workload_change_trigger(
                            new_count / store_size_before)
                    except Exception:
                        pass

            # Real indexes on the target, read per request; arms covered by an
            # existing index (or its leftmost prefix) are masked before the
            # bandit sees them.
            existing_prefixes = self._existing_prefixes(dsn)

            if recommend_only:
                chosen_names = self._select_without_rounds(
                    eval_conn, meta, cfg, list(self._query_store.values()), notes,
                    existing_prefixes, max_memory)
                rounds_executed = 0
                baseline_cost = None
                final_cost = None
                self._counters["recommend_only_requests"] += 1
            else:
                result = self._run_rounds(eval_conn, meta, cfg, queries, deadline,
                                          cfg["rounds"], existing_prefixes)
                chosen_names = result["best_names"]
                baseline_cost = result.get("baseline_cost")
                final_cost = result.get("final_cost")
                rounds_executed = result["rounds_executed"]
                notes.extend(result["notes"])
                self._counters["rounds_total"] += rounds_executed
                self._counters["observations_since_training"] += 1
                if result["time_budget_hit"]:
                    notes.append("time budget reached; returning the best configuration "
                                 "observed so far")

            # translate chosen arms into DDL
            # _to_ddl appends its own notes into the shared list; the
            # return value is the (budget-enforced) record list only
            rec_list = self._to_ddl(chosen_names, dsn, eval_conn, meta, budget, notes)

            impact = None
            if not recommend_only and rec_list:
                impact = self._estimate_impact(
                    eval_conn, [q for q in queries],
                    [bandit_helper.bandit_arm_store[n] for n in chosen_names
                     if n in bandit_helper.bandit_arm_store], deadline, notes)

            apply_results = None
            if apply_flag and rec_list:
                apply_results = self._apply_indexes(rec_list, dsn)

            self._counters["requests_served"] += 1
            self._counters["queries_analyzed_total"] += len(queries)
            self._last_config = cfg
            self._persist_state()
            if not recommend_only:
                self._maybe_spawn_training(eval_dsn, cfg)

            response = {
                "recommended_indexes": rec_list,
                "metadata": {
                    "strategy_type": "learned (C3UCB combinatorial contextual bandit)",
                    "optimization_time": round(time.time() - t0, 3),
                    "estimated_impact": impact,
                    "num_queries_analyzed": len(queries),
                    "budget_used": self._budget_used(budget, rec_list, max_memory),
                    "eval_mode": "hypothetical" if eval_dsn == dsn else "standby",
                    "mode": "inference-only" if recommend_only else "online-training",
                    "rounds_executed": rounds_executed,
                    "workload_queries_total": len(entries),
                    "queries_known": len(self._query_store),
                    "arms_known": len(bandit_helper.bandit_arm_store),
                },
            }
            if apply_results is not None:
                response["apply_results"] = apply_results
            if notes:
                response["metadata"]["degrade_note"] = "; ".join(notes)
            if recommend_only and impact is None:
                notes.append("estimated_impact omitted in inference-only mode "
                             "(no what-if execution)")
                response["metadata"]["degrade_note"] = "; ".join(notes)
            if not rec_list and not notes:
                response["metadata"]["degrade_note"] = (
                    "no beneficial index configuration found within the budget")
            print(f"[req] {time.time() - t0:.2f}s recs={len(rec_list)} "
                  f"analyzed={len(queries)} rounds={rounds_executed}", flush=True)
            return response
        finally:
            if eval_conn is not None:
                try:
                    cur = eval_conn.cursor()
                    cur.execute("SELECT hypopg_reset_index()")
                    cur.fetchone()
                except Exception:
                    pass
                try:
                    eval_conn.close()
                except Exception:
                    pass
            _HYP_NAME_MAP.clear()
            _HYP_OID_MAP.clear()

    # -- helpers -------------------------------------------------------------

    def _parse_config(self, config_field):
        cfg = dict(DEFAULT_CONFIG)
        if not config_field:
            return cfg
        if isinstance(config_field, str):
            try:
                config_field = json.loads(config_field)
            except Exception:
                config_field = {"model_dir": config_field}
        if isinstance(config_field, dict):
            converters = {
                "rounds": int, "train_trigger": int, "training_rounds": int,
                "training_workloads": int, "training_workload_size": int,
                "training_sample_seed": int,
                "time_budget_seconds": float, "alpha": float, "lambda": float,
                "max_index_width": int, "statement_timeout_ms": int,
                "probe_timeout_ms": int, "max_query_store": int,
                "max_arm_store": int, "max_state_mb": float,
            }
            for k, conv in converters.items():
                if k in config_field and config_field[k] is not None:
                    try:
                        cfg[k] = conv(config_field[k])
                    except (TypeError, ValueError):
                        pass
            if "model_dir" in config_field and config_field["model_dir"]:
                md = str(config_field["model_dir"])
                cfg["model_dir"] = md if os.path.isabs(md) else os.path.join(REPO, md)
        return cfg

    def _parse_budget(self, budget_field):
        budget = {"max_indexes": DEFAULT_MAX_INDEXES}
        if not budget_field:
            return budget
        if isinstance(budget_field, str):
            try:
                budget_field = json.loads(budget_field)
            except Exception:
                return budget
        if isinstance(budget_field, dict):
            for k in ("storage_mb", "max_indexes", "max_write_overhead_pct"):
                if budget_field.get(k) is not None:
                    try:
                        budget[k] = float(budget_field[k])
                    except (TypeError, ValueError):
                        pass
        return budget

    def _build_queries(self, eval_conn, entries, meta):
        """Query objects (selectivity via cost-only EXPLAIN)."""
        queries, skipped = [], []
        for e in entries:
            if e.get("unparsed"):
                skipped.append("query skipped: SQL not parseable by the workload parser")
                continue
            qid = int(hashlib.md5(e["sql"].encode("utf-8")).hexdigest()[:12], 16)
            if qid in self._query_store:
                q = self._query_store[qid]
                q.query_string = e["sql"]
                q.frequency += 1
                q.last_request_at = time.time()
                queries.append(q)
                continue
            try:
                q = Query(eval_conn, qid, e["sql"], e["predicates"] or {}, e["payload"] or {}, 0)
            except _QueryPlanError as ex:
                reason = str(ex).splitlines()[0] if str(ex) else "plan unavailable"
                skipped.append(f"planning failed ({reason})")
                continue
            except Exception as ex:
                skipped.append(f"unparseable query ({type(ex).__name__})")
                continue
            q.last_request_at = time.time()
            if e.get("weight"):
                try:
                    q.frequency = max(1, int(e["weight"]))
                except (TypeError, ValueError):
                    pass
            queries.append(q)
        return queries, skipped

    def _ensure_bandit(self, cfg, max_memory):
        """Reuse the in-memory bandit across requests (continuity); rebuild
        it only after a disk reload, a state reset, or a width change. The
        per-request budget enters through the oracle, which is swapped on
        every call."""
        context_size = self._live_context_size
        live = self._bandit
        if (live is not None
                and getattr(live, "context_size", None) == context_size
                and getattr(live, "_skill_epoch", None) == self._bandit_epoch
                and live._skill_n_columns == self._live_n_columns):
            live.oracle = Oracle(max_memory)
            return
        state = self._bandit_state
        restore = (state is not None
                   and state["context_size"] == context_size
                   and state["n_columns"] == self._live_n_columns)
        self._bandit = bandit_module.C3UCB(
            context_size, float(cfg["alpha"]), float(cfg["lambda"]),
            Oracle(max_memory))
        self._bandit._skill_epoch = self._bandit_epoch
        self._bandit._skill_n_columns = self._live_n_columns
        if restore:
            self._bandit.v = numpy.array(state["v"], dtype=float)
            self._bandit.b = numpy.array(state["b"], dtype=float)
            self._bandit.hyper_alpha = float(state["hyper_alpha"])
        else:
            # fresh bootstrap: the delta base is the pristine (lambda*I, 0)
            # matrices, and any stale on-disk state (other schema width) is
            # replaced rather than merged (the context-size guard in
            # _persist_state rejects a mismatched merge).
            self._bandit_state = None
            self._state_base = {
                "v": numpy.array(self._bandit.v, dtype=float),
                "b": numpy.array(self._bandit.b, dtype=float),
                "alpha": float(self._bandit.hyper_alpha),
            }
            self._base_rev = 0

    def _run_rounds(self, eval_conn, meta, cfg, queries, deadline, n_rounds,
                    existing_prefixes=frozenset(), pool=None):
        """What-if round loop; every round runs on the hypothetical (HypoPG) path."""
        notes = []
        bandit = self._bandit
        all_columns = meta["all_columns"]
        n_cols = meta["n_columns"]
        chosen_arms_last_round = {}
        super_arm_scores, super_arm_counts = {}, {}
        best_names = frozenset()
        baseline_cost = None
        final_cost = None
        rounds_executed = 0
        time_budget_hit = False
        wide_filtered = 0
        existing_masked = 0
        max_width = max(0, int(cfg.get("max_index_width") or 0))
        watch_last = time.time()

        for t in range(int(n_rounds)):
            if time.time() > deadline:
                time_budget_hit = True
                break
            # Per-round query bookkeeping (frequency/last_seen).
            for q in queries:
                q.last_seen = t
            query_obj_list_past = []
            source = self._query_store.values() if pool is None else pool
            for q in source:
                if t - q.last_seen <= constants.QUERY_MEMORY and 0 <= q.first_seen < t:
                    query_obj_list_past.append(q)

            # Arm generation from past queries, with clustered_index_time /
            # query_ids bookkeeping. max_index_width constrains key columns
            # (0 = unrestricted); INCLUDE payload columns do not count.
            index_arms = {}
            for q in query_obj_list_past:
                try:
                    bandit_arms_tmp = bandit_helper.gen_arms_from_predicates_v2(eval_conn, q)
                except Exception as ex:
                    if f"query arm generation degraded ({type(ex).__name__})" not in notes:
                        notes.append(f"query arm generation degraded ({type(ex).__name__})")
                    continue
                for key, index_arm in bandit_arms_tmp.items():
                    if max_width and len(index_arm.index_cols) > max_width:
                        wide_filtered += 1
                        continue
                    arm_key = (index_arm.table_name.lower(),
                               tuple(c.lower() for c in index_arm.index_cols))
                    if arm_key in existing_prefixes:
                        existing_masked += 1
                        continue
                    if key not in index_arms:
                        index_arm.query_ids = set()
                        index_arm.query_ids_backup = set()
                        index_arm.clustered_index_time = 0
                        index_arms[key] = index_arm
                    scan_times = q.table_scan_times.get(index_arm.table_name) or []
                    index_arms[key].clustered_index_time += max(scan_times) if scan_times else 0
                    index_arms[key].query_ids.add(index_arm.query_id)
                    index_arms[key].query_ids_backup.add(index_arm.query_id)

            index_arm_list = list(index_arms.values())
            bandit.set_arms(index_arm_list)
            bandit.upper_bounds = []  # select_arm_v2 appends per call

            try:
                context_vectors_v1 = bandit_helper.get_name_encode_context_vectors_v2(
                    index_arms, all_columns, n_cols,
                    constants.CONTEXT_UNIQUENESS, constants.CONTEXT_INCLUDES)
                context_vectors_v2 = bandit_helper.get_derived_value_context_vectors_v3(
                    eval_conn, index_arms, query_obj_list_past,
                    chosen_arms_last_round, not constants.CONTEXT_INCLUDES)
            except Exception as ex:
                notes.append(f"context vector construction degraded ({type(ex).__name__})")
                break
            context_vectors = []
            for i in range(len(index_arm_list)):
                context_vectors.append(
                    numpy.array(list(context_vectors_v2[i]) + list(context_vectors_v1[i]),
                                ndmin=2))

            chosen_arm_ids = bandit.select_arm_v2(context_vectors, t)

            chosen_arms = {}
            for i in chosen_arm_ids:
                arm = index_arm_list[i]
                chosen_arms[arm.index_name] = arm

            keys_last_round = set(chosen_arms_last_round.keys())
            keys_this_round = set(chosen_arms.keys())
            key_intersection = keys_last_round & keys_this_round
            key_additions = keys_this_round - key_intersection
            key_deletions = keys_last_round - key_intersection
            added_arms = {k: chosen_arms[k] for k in key_additions}
            deleted_arms = {k: chosen_arms_last_round[k] for k in key_deletions}

            try:
                execute_cost, creation_cost, arm_rewards = \
                    sql_helper.hyp_create_query_drop_v2(
                        eval_conn, "public", index_arms, added_arms, deleted_arms,
                        query_obj_list_past)
            except Exception as ex:
                notes.append(f"what-if round degraded after {rounds_executed} rounds "
                             f"({type(ex).__name__})")
                chosen_arms_last_round = chosen_arms
                break
            if t == 0:
                baseline_cost = execute_cost  # round 0 has no arms: pure baseline

            bandit.update_v4(chosen_arm_ids, arm_rewards)

            super_arm_id = frozenset(chosen_arms.keys())
            if super_arm_id in super_arm_scores:
                super_arm_scores[super_arm_id] = (
                    super_arm_scores[super_arm_id] * super_arm_counts[super_arm_id]
                    + execute_cost)
                super_arm_counts[super_arm_id] += 1
                super_arm_scores[super_arm_id] /= super_arm_counts[super_arm_id]
            else:
                super_arm_counts[super_arm_id] = 1
                super_arm_scores[super_arm_id] = execute_cost
            if super_arm_scores:
                best_names = min(super_arm_scores, key=super_arm_scores.get)

            chosen_arms_last_round = chosen_arms
            rounds_executed += 1
            now = time.time()
            if now - watch_last >= 15:
                print(f"[watch] round {t}/{n_rounds} arms={len(index_arm_list)} "
                      f"chosen={len(chosen_arms)} exec_cost={execute_cost:.1f}",
                      flush=True)
                watch_last = now
            if now > deadline:
                time_budget_hit = True
                break

        if not best_names and chosen_arms_last_round:
            best_names = frozenset(chosen_arms_last_round.keys())
        if wide_filtered:
            notes.append(f"{wide_filtered} candidate arms wider than "
                         f"max_index_width={max_width} excluded")
        if existing_masked:
            # expected behavior (the index already exists), not a degradation
            print(f"[mask] {existing_masked} candidate arms masked: covered by "
                  "existing indexes", flush=True)
        if best_names and super_arm_scores and best_names in super_arm_scores:
            final_cost = super_arm_scores[best_names]
        return {
            "best_names": best_names,
            "baseline_cost": baseline_cost,
            "final_cost": final_cost,
            "rounds_executed": rounds_executed,
            "notes": notes,
            "time_budget_hit": time_budget_hit,
        }

    def _select_without_rounds(self, eval_conn, meta, cfg, queries, notes,
                               existing_prefixes=frozenset(), max_memory=None):
        """--recommend-only: super-arm selection from the learned state with
        no what-if rounds and no bandit update."""
        bandit = self._bandit
        bandit.oracle = Oracle(float(max_memory if max_memory is not None
                                     else DEFAULT_STORAGE_MB))
        max_width = max(0, int(cfg.get("max_index_width") or 0))
        existing_masked = 0
        index_arms = {}
        for q in queries:
            try:
                tmp = bandit_helper.gen_arms_from_predicates_v2(eval_conn, q)
            except Exception as ex:
                if f"query arm generation degraded ({type(ex).__name__})" not in notes:
                    notes.append(f"query arm generation degraded ({type(ex).__name__})")
                continue
            for key, arm in tmp.items():
                if max_width and len(arm.index_cols) > max_width:
                    continue
                arm_key = (arm.table_name.lower(),
                           tuple(c.lower() for c in arm.index_cols))
                if arm_key in existing_prefixes:
                    existing_masked += 1
                    continue
                if key not in index_arms:
                    index_arms[key] = arm
        if existing_masked:
            # expected behavior (the index already exists), not a degradation
            print(f"[mask] {existing_masked} candidate arms masked: covered by "
                  "existing indexes", flush=True)
        index_arm_list = list(index_arms.values())
        bandit.set_arms(index_arm_list)
        bandit.upper_bounds = []
        try:
            cv1 = bandit_helper.get_name_encode_context_vectors_v2(
                index_arms, meta["all_columns"], meta["n_columns"],
                constants.CONTEXT_UNIQUENESS, constants.CONTEXT_INCLUDES)
            cv2 = bandit_helper.get_derived_value_context_vectors_v3(
                eval_conn, index_arms, queries, {}, not constants.CONTEXT_INCLUDES)
            context_vectors = [
                numpy.array(list(cv2[i]) + list(cv1[i]), ndmin=2)
                for i in range(len(index_arm_list))]
            chosen_arm_ids = bandit.select_arm_v2(context_vectors, 1)
        except Exception as ex:
            notes.append(f"inference-only selection degraded ({type(ex).__name__})")
            return frozenset()
        return frozenset(index_arm_list[i].index_name for i in chosen_arm_ids)

    # -- output assembly -----------------------------------------------------

    def _existing_indexes(self, dsn):
        """Real indexes on the MAIN dsn (re-read per request — the wrapper's
        own --apply or external DDL can change this set mid-session)."""
        names, cols = set(), set()
        try:
            conn = _pg_connect(dsn, 30000)
        except Exception:
            return names, cols
        try:
            cur = conn.cursor()
            cur.execute("SELECT schemaname, tablename, indexname, indexdef "
                        "FROM pg_indexes WHERE schemaname NOT IN ('pg_catalog','information_schema')")
            for schema, table, name, indexdef in cur.fetchall():
                names.add(name)
                cseq = _parse_indexdef_key_columns(indexdef)
                if cseq:
                    cols.add((table, tuple(cseq)))
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return names, cols

    def _existing_prefixes(self, dsn):
        """Leftmost prefixes of the real indexes on the target.

        A btree also serves leftmost-prefix predicates, so a candidate whose
        key columns match an existing prefix is masked before the bandit sees
        it. Read per request (--apply and external DDL mutate this set).
        """
        try:
            _names, cols = self._existing_indexes(dsn)
        except Exception:
            return set()
        prefixes = set()
        for table, cseq in cols:
            for k in range(1, len(cseq) + 1):
                prefixes.add((table.lower(), tuple(cseq[:k])))
        return prefixes

    def _to_ddl(self, chosen_names, dsn, eval_conn, meta, budget, notes):
        arm_store = bandit_helper.bandit_arm_store
        arms = [arm_store[n] for n in chosen_names if n in arm_store]
        if chosen_names and not arms:
            notes.append("chosen arms could not be resolved to index specifications")
        existing_names, existing_cols = self._existing_indexes(dsn)

        rec_list = []
        seen = set()
        for arm in arms:
            table = arm.table_name
            cols = [c for c in arm.index_cols]
            if table not in meta["tables"]:
                notes.append(f"index on unknown table {table} skipped")
                continue
            if arm.memory is None or float(arm.memory) >= 99999999:
                notes.append(f"index {arm.index_name} skipped: index key too wide to build "
                             "(1700-byte key guard)")
                continue
            low_cols = [c.lower() for c in cols]
            redundant = False
            for et, ec in existing_cols:
                if et == table and list(ec[:len(low_cols)]) == low_cols:
                    redundant = True
                    break
            if redundant:
                notes.append(f"index on ({table}, {', '.join(cols)}) skipped: covered by "
                             "an existing index")
                continue
            info = meta["tables"][table]
            schema = info["schema"]
            # PostgreSQL truncates identifiers at 63 bytes; keep response
            # name == DB name.
            index_name = arm.index_name[:63]
            suffix = 0
            while index_name in existing_names or index_name in seen:
                suffix += 1
                index_name = f"{arm.index_name[:60]}_{suffix}"
            seen.add(index_name)
            col_sql = ", ".join(f'"{c}"' for c in cols)
            ddl = f'CREATE INDEX "{index_name}" ON "{schema}"."{table}" ({col_sql})'
            if arm.include_cols:
                incl_sql = ", ".join(f'"{c}"' for c in arm.include_cols)
                ddl += f" INCLUDE ({incl_sql})"
            table_mb = max(1.0, info.get("size_bytes", 0) / (1024.0 * 1024.0))
            storage_mb = round(float(arm.memory), 3)
            write_overhead = round(min(100.0, 1.0 + (storage_mb / table_mb) * 5.0), 3)
            rec_list.append({
                "ddl": ddl,
                "table": table,
                "columns": cols,
                "index_name": index_name,
                "storage_mb": storage_mb,
                "build_time_estimate": round(info["row_count"] / 200000.0, 3),
                "write_overhead_estimate": write_overhead,
            })

        # NOTE: appends into `notes` (shared with the response); the record
        # list is the return value.
        # budget enforcement (storage is already enforced inside the oracle
        # via max_memory; count and write overhead are enforced here)
        max_indexes = budget.get("max_indexes")
        if max_indexes is not None and len(rec_list) > int(max_indexes):
            dropped = len(rec_list) - int(max_indexes)
            rec_list = rec_list[: int(max_indexes)]
            notes.append(f"{dropped} indexes dropped to respect the max_indexes budget")
        storage_mb = budget.get("storage_mb")
        if storage_mb is not None:
            total, kept = 0.0, []
            for r in rec_list:
                if total + (r.get("storage_mb") or 0) > storage_mb:
                    continue
                total += r.get("storage_mb") or 0
                kept.append(r)
            if len(kept) != len(rec_list):
                notes.append(f"{len(rec_list) - len(kept)} indexes dropped to respect "
                             "the storage_mb budget")
            rec_list = kept
        max_wo = budget.get("max_write_overhead_pct")
        if max_wo is not None:
            kept = [r for r in rec_list
                    if (r.get("write_overhead_estimate") or 0) <= float(max_wo)]
            if len(kept) != len(rec_list):
                notes.append(f"{len(rec_list) - len(kept)} indexes dropped to respect "
                             "the max_write_overhead_pct budget")
            rec_list = kept
        return rec_list

    def _estimate_impact(self, eval_conn, queries, arms, deadline, notes):
        """Estimated workload cost reduction of the recommended configuration
        (cost-only EXPLAIN, hypothetical indexes)."""
        if not queries or not arms:
            return None
        if time.time() > deadline - 5:
            notes.append("impact estimation skipped (time budget)")
            return None
        cur = eval_conn.cursor()
        try:
            cur.execute("SELECT hypopg_reset_index()")
            cur.fetchone()
            _patched_hyp_bulk_create_indexes(eval_conn, "public",
                                             {a.index_name: a for a in arms})
            final_cost = 0.0
            for q in queries:
                try:
                    final_cost += float(_explain_json(cur, q.query_string).get("Total Cost") or 0)
                except Exception:
                    pass
            cur.execute("SELECT hypopg_reset_index()")
            cur.fetchone()
            base_cost = 0.0
            for q in queries:
                try:
                    base_cost += float(_explain_json(cur, q.query_string).get("Total Cost") or 0)
                except Exception:
                    pass
            if base_cost <= 0:
                return None
            return round((base_cost - final_cost) / base_cost * 100.0, 2)
        except Exception as ex:
            notes.append(f"impact estimation degraded ({type(ex).__name__})")
            return None
        finally:
            try:
                cur.execute("SELECT hypopg_reset_index()")
                cur.fetchone()
            except Exception:
                pass

    def _apply_indexes(self, rec_list, dsn):
        """The ONLY path that mutates the main database."""
        results = []
        try:
            conn = _pg_connect(dsn, 600000)
        except Exception as e:
            return [{"index_name": r["index_name"], "status": "error",
                     "error": f"cannot connect: {type(e).__name__}"} for r in rec_list]
        try:
            cur = conn.cursor()
            for r in rec_list:
                try:
                    cur.execute(r["ddl"])
                    results.append({"index_name": r["index_name"], "status": "created",
                                    "error": None})
                except Exception as e:
                    results.append({"index_name": r["index_name"], "status": "error",
                                    "error": str(e).splitlines()[0][:200] if str(e) else
                                             type(e).__name__})
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return results

    def _budget_used(self, budget, rec_list, effective_storage_mb=None):
        consumed_storage = round(sum(r.get("storage_mb") or 0.0 for r in rec_list), 3)
        consumed_wo = round(max((r.get("write_overhead_estimate") or 0.0) for r in rec_list), 3) \
            if rec_list else 0.0
        return {
            "storage_mb": (budget.get("storage_mb")
                           if budget.get("storage_mb") is not None
                           else effective_storage_mb),
            "max_indexes": budget.get("max_indexes"),
            "max_write_overhead_pct": budget.get("max_write_overhead_pct"),
            "consumed": {
                "storage_mb": consumed_storage,
                "indexes": len(rec_list),
                "write_overhead_pct": consumed_wo,
            },
        }

    def _degrade(self, reason, cfg, num_analyzed, notes, eval_mode="hypothetical",
                 recommend_only=False, budget=None):
        note = reason if not notes else "; ".join([reason] + list(notes))
        return {
            "recommended_indexes": [],
            "metadata": {
                "strategy_type": "learned (C3UCB combinatorial contextual bandit)",
                "optimization_time": 0.0,
                "num_queries_analyzed": num_analyzed,
                "budget_used": self._budget_used(budget or {}, [], DEFAULT_STORAGE_MB),
                "eval_mode": eval_mode,
                "mode": "inference-only" if recommend_only else "online-training",
                "degrade_note": note,
            },
        }

    # -- introspection / lifecycle -------------------------------------------

    def state(self):
        with self._lock:
            counters = dict(self._counters)
            cfg = dict(self._last_config)
            training = self._training_alive()
        return {
            "idle": not training and not self._shutdown,
            "strategy": "DBA bandits (C3UCB)",
            "requests_served": counters["requests_served"],
            "rounds_total": counters["rounds_total"],
            "queries_analyzed_total": counters["queries_analyzed_total"],
            "recommend_only_requests": counters["recommend_only_requests"],
            "training": {
                "in_progress": training,
                "runs_completed": counters["training_runs"],
                "observations_since_training": counters["observations_since_training"],
                "last_sample_seed": counters.get("last_training_sample_seed"),
                "last_pool_sizes": counters.get("last_training_pool_sizes"),
            },
            "bandit": {
                "context_size": self._context_size(),
            },
            "state": {
                "query_store": len(self._query_store),
                "arm_store": len(bandit_helper.bandit_arm_store),
                "last_state_bytes": self._last_state_bytes,
                "last_evicted_queries": self._last_evicted_queries,
                "last_evicted_arms": self._last_evicted_arms,
                "state_over_budget": self._state_over_budget,
            },
            "active_config": cfg,
            "state_dir": self.state_dir,
            "strategy_type": "learned (C3UCB combinatorial contextual bandit)",
        }

    def shutdown(self):
        self._shutdown = True
        with self._lock:
            self._persist_state()
        pid = self._training_pid
        if pid is not None:
            # bounded wait: let an in-flight training run finish its atomic save
            deadline = time.time() + 60
            while time.time() < deadline and self._training_alive():
                time.sleep(1.0)
            if self._training_alive():
                try:
                    os.kill(pid, 15)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
_SERVER = {}


from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler as _BHR  # noqa: E402


def _make_handler(server_obj):
    class Handler(_BHR):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # client disconnected (timeout/cancel); the request itself
                # already completed — not a server-side failure
                pass

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, {"status": "ok", "strategy": "DBA bandits"})
                return
            if self.path.startswith("/state"):
                self._send(200, server_obj.state())
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.startswith("/recommend"):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    req = json.loads(raw.decode("utf-8") or "{}")
                except Exception as e:
                    self._send(200, server_obj._degrade(f"invalid JSON body: {e}", {}, 0, []))
                    return
                t0 = time.time()
                try:
                    resp = server_obj.recommend(req)
                except Exception as e:
                    traceback.print_exc(file=_debug_stream())
                    resp = server_obj._degrade(
                        f"internal degrade: {type(e).__name__}", {}, 0, [],
                        recommend_only=bool(req.get("recommend_only")),
                        budget=server_obj._parse_budget(req.get("budget")))
                self._send(200, resp)
                return
            if self.path.startswith("/shutdown"):
                self._send(200, {"status": "shutting down"})
                threading.Thread(target=server_obj.shutdown, daemon=True).start()
                threading.Thread(
                    target=lambda: (time.sleep(0.5), _SERVER["httpd"].shutdown()),
                    daemon=True).start()
                return
            self._send(404, {"error": "not found"})

    return Handler


def _debug_stream():
    """Tracebacks go to a file, not stdout."""
    try:
        os.makedirs(_STATE_DIR_ENV, exist_ok=True)
        return open(os.path.join(_STATE_DIR_ENV, "error.log"), "a")
    except Exception:
        import io
        return io.StringIO()


def main():
    args = sys.argv[1:]
    if "--train" in args:
        model_dir = DEFAULT_STATE_DIR
        eval_dsn = None
        rounds = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg in ("--model-dir", "--eval-dsn", "--rounds") and i + 1 < len(args):
                val = args[i + 1]
                if arg == "--model-dir":
                    model_dir = val
                elif arg == "--eval-dsn":
                    eval_dsn = val
                else:
                    try:
                        rounds = int(val)
                    except ValueError:
                        rounds = None
                i += 2
            else:
                i += 1
        if not eval_dsn:
            print("[train] --eval-dsn required", flush=True)
            sys.exit(2)
        server = DBABanditsServer(state_dir=model_dir)
        server.train_once(eval_dsn, rounds=rounds)
        _atomic_write(os.path.join(model_dir, _TRAINING_FILE),
                      json.dumps({"pid": os.getpid(), "status": "done",
                                  "finished_at": time.time()}))
        return

    port = int(os.environ.get("PORT", "8765"))
    state_dir = os.environ.get("DBABANDITS_STATE_DIR", DEFAULT_STATE_DIR)
    server_obj = DBABanditsServer(state_dir=state_dir)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(server_obj))
    httpd.daemon_threads = True
    _SERVER["httpd"] = httpd
    print(f"DBA bandits wrapper listening on {port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        server_obj.shutdown()


if __name__ == "__main__":
    main()
