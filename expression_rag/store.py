"""Append-only expression storage and isolated embedding spaces.

This database belongs to this plugin.  No MaiBot tables or original corpus
files are read or changed by the storage layer.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import hashlib
import json
import math
import sqlite3
import unicodedata


MAX_IMPORT_ENTRIES = 1000
MAX_VECTOR_DIMENSION = 32768
MAX_VECTOR_BATCH = 1000
MAX_SOURCE_CHARS = 240
_SEED_MARKER = "initial_seed_complete"


def _text(value: object, name: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须是字符串")
    result = unicodedata.normalize("NFC", value).strip()
    if not result or len(result) > max_chars:
        raise ValueError(f"{name} 不能为空，且最多 {max_chars} 个字符")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in result):
        raise ValueError(f"{name} 不能含有控制字符")
    return result


def _namespace(value: object) -> str:
    return _text(value, "向量空间 namespace", 512)


def _vector(value: object, label: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_VECTOR_DIMENSION:
        raise ValueError(f"{label} 必须是含 1–{MAX_VECTOR_DIMENSION} 个数字的向量")
    result: List[float] = []
    for number in value:
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"{label} 的所有分量必须是有限数字")
        try:
            converted = float(number)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{label} 的所有分量必须是有限数字") from exc
        if not math.isfinite(converted):
            raise ValueError(f"{label} 不能包含 NaN 或 Infinity")
        result.append(converted)
    if not any(number != 0 for number in result):
        raise ValueError(f"{label} 不能是零向量")
    return result


def _entry(value: object, default_source: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("每条表达必须是 JSON 对象")
    situation = _text(value.get("situation"), "situation", 400)
    style = _text(value.get("style"), "style", 1200)
    examples = value.get("content_list")
    if not isinstance(examples, list) or not 1 <= len(examples) <= 20:
        raise ValueError("content_list 必须包含 1–20 条参考句")
    content_list = [_text(item, "content_list 参考句", 1000) for item in examples]
    source = _text(value.get("source", default_source), "source", MAX_SOURCE_CHARS)
    # Source labels and import order do not change an expression's identity.
    canonical = json.dumps(
        {"situation": situation, "style": style, "content_list": content_list},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    identifier = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "id": identifier,
        "situation": situation,
        "style": style,
        "content_list": content_list,
        "source": source,
    }


def _entries(values: object, source: str) -> List[Dict[str, Any]]:
    if not isinstance(values, list) or len(values) > MAX_IMPORT_ENTRIES:
        raise ValueError(f"一次导入必须为列表，且最多 {MAX_IMPORT_ENTRIES} 条表达")
    _text(source, "source", MAX_SOURCE_CHARS)
    # Validate the entire batch before performing any writes.
    return [_entry(value, source) for value in values]


class ExpressionStore:
    """SQLite storage with one short-lived connection per synchronous call."""

    def __init__(self, path: object) -> None:
        if str(path) == ":memory:":
            raise ValueError("ExpressionStore 需要持久化数据库文件路径")
        self.path = Path(path)  # type: ignore[arg-type]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as database:
            database.executescript(
                """
                CREATE TABLE IF NOT EXISTS expression_entries (
                    id TEXT PRIMARY KEY,
                    situation TEXT NOT NULL,
                    style TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS expression_metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embedding_spaces (
                    namespace TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL CHECK (dimension > 0)
                );
                CREATE TABLE IF NOT EXISTS expression_vectors (
                    namespace TEXT NOT NULL,
                    entry_id TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    PRIMARY KEY (namespace, entry_id),
                    FOREIGN KEY (namespace) REFERENCES embedding_spaces(namespace),
                    FOREIGN KEY (entry_id) REFERENCES expression_entries(id)
                );
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        database = sqlite3.connect(str(self.path), timeout=15)
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA foreign_keys=ON")
        try:
            with database:
                yield database
        finally:
            database.close()

    @staticmethod
    def _insert(database: sqlite3.Connection, entries: List[Dict[str, Any]]) -> int:
        added = 0
        for entry in entries:
            cursor = database.execute(
                """INSERT INTO expression_entries
                   (id, situation, style, content_json, source) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO NOTHING""",
                (
                    entry["id"],
                    entry["situation"],
                    entry["style"],
                    json.dumps(entry["content_list"], ensure_ascii=False),
                    entry["source"],
                ),
            )
            added += cursor.rowcount
        return added

    def initialize(self, seed_entries: List[Dict[str, Any]]) -> None:
        """Seed once, transactionally; later starts never restore old seed data."""
        with self._connection() as database:
            database.execute("BEGIN IMMEDIATE")
            complete = database.execute(
                "SELECT value_json FROM expression_metadata WHERE key=?", (_SEED_MARKER,)
            ).fetchone()
            if complete is not None:
                return
            seeds = _entries(seed_entries, "精选语料")
            self._insert(database, seeds)
            database.execute(
                "INSERT INTO expression_metadata (key, value_json) VALUES (?, ?)",
                (_SEED_MARKER, "true"),
            )

    def entries(self) -> List[Dict[str, Any]]:
        with self._connection() as database:
            rows = database.execute(
                "SELECT id, situation, style, content_json, source "
                "FROM expression_entries ORDER BY rowid"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "situation": row["situation"],
                "style": row["style"],
                "content_list": json.loads(row["content_json"]),
                "source": row["source"],
            }
            for row in rows
        ]

    def import_entries(
        self, items: List[Dict[str, Any]], source: str = "WebUI导入"
    ) -> Dict[str, int]:
        entries = _entries(items, source)
        with self._connection() as database:
            database.execute("BEGIN IMMEDIATE")
            added = self._insert(database, entries)
            total = int(database.execute("SELECT COUNT(*) FROM expression_entries").fetchone()[0])
        return {"added": added, "skipped": len(entries) - added, "total": total}

    def get_meta(self, key: str, default: Any = None) -> Any:
        key = _text(key, "元数据 key", 200)
        with self._connection() as database:
            row = database.execute(
                "SELECT value_json FROM expression_metadata WHERE key=?", (key,)
            ).fetchone()
        return default if row is None else json.loads(row[0])

    def set_meta(self, key: str, value: Any) -> None:
        key = _text(key, "元数据 key", 200)
        if key == _SEED_MARKER:
            raise ValueError("初始化标记由存储层维护，不能手动修改")
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("元数据 value 必须是有效的 JSON 数据") from exc
        if len(encoded.encode("utf-8")) > 1_000_000:
            raise ValueError("元数据 value 不能超过 1 MB")
        with self._connection() as database:
            database.execute(
                """INSERT INTO expression_metadata (key, value_json) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json""",
                (key, encoded),
            )

    def vectors(self, namespace: str) -> Dict[str, List[float]]:
        namespace = _namespace(namespace)
        with self._connection() as database:
            rows = database.execute(
                "SELECT entry_id, vector_json FROM expression_vectors WHERE namespace=?",
                (namespace,),
            ).fetchall()
        return {row["entry_id"]: _vector(json.loads(row["vector_json"]), "已保存向量") for row in rows}

    def put_vectors(self, namespace: str, items: List[Tuple[str, List[float]]]) -> None:
        namespace = _namespace(namespace)
        if not isinstance(items, list) or len(items) > MAX_VECTOR_BATCH:
            raise ValueError(f"向量批次必须为列表，且最多 {MAX_VECTOR_BATCH} 条")
        if not items:
            return
        validated: List[Tuple[str, List[float]]] = []
        identifiers = set()
        dimension: Optional[int] = None
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("向量批次的每项必须是 (表达 id, 向量)")
            identifier = _text(item[0], "表达 id", 128)
            if identifier in identifiers:
                raise ValueError("同一向量批次不能包含重复的表达 id")
            identifiers.add(identifier)
            vector = _vector(item[1], f"表达 {identifier} 的向量")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("同一 namespace 的向量维度必须一致")
            validated.append((identifier, vector))

        with self._connection() as database:
            database.execute("BEGIN IMMEDIATE")
            row = database.execute(
                "SELECT dimension FROM embedding_spaces WHERE namespace=?", (namespace,)
            ).fetchone()
            if row is not None and row[0] != dimension:
                raise ValueError(f"向量维度不一致：该 namespace 为 {row[0]} 维，新向量为 {dimension} 维")
            for identifier, _ in validated:
                if database.execute(
                    "SELECT 1 FROM expression_entries WHERE id=?", (identifier,)
                ).fetchone() is None:
                    raise ValueError(f"表达 id 不存在：{identifier}")
            database.execute(
                "INSERT INTO embedding_spaces(namespace, dimension) VALUES (?, ?) "
                "ON CONFLICT(namespace) DO NOTHING",
                (namespace, dimension),
            )
            database.executemany(
                """INSERT INTO expression_vectors(namespace, entry_id, vector_json) VALUES (?, ?, ?)
                   ON CONFLICT(namespace, entry_id) DO UPDATE SET vector_json=excluded.vector_json""",
                [
                    (namespace, identifier, json.dumps(vector, allow_nan=False))
                    for identifier, vector in validated
                ],
            )

    def index_status(self, namespace: str) -> Dict[str, int]:
        namespace = _namespace(namespace)
        with self._connection() as database:
            total = int(database.execute("SELECT COUNT(*) FROM expression_entries").fetchone()[0])
            indexed = int(
                database.execute(
                    "SELECT COUNT(*) FROM expression_vectors WHERE namespace=?", (namespace,)
                ).fetchone()[0]
            )
            row = database.execute(
                "SELECT dimension FROM embedding_spaces WHERE namespace=?", (namespace,)
            ).fetchone()
        return {
            "total": total,
            "indexed": indexed,
            "missing": total - indexed,
            "dimension": int(row[0]) if row is not None else 0,
        }
