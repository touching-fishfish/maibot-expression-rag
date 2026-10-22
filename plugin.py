"""Style examples retrieved before a reply, independent of sealed memories."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import asyncio
import hashlib
import json
import time

from maibot_sdk import API, HookHandler, MaiBotPlugin
from maibot_sdk.types import HookMode, HookOrder

from .expression_rag.config import ExpressionRAGConfig
from .expression_rag.embedding import EmbeddingClient, EmbeddingError, settings_signature, vector_namespace
from .expression_rag.emoji import convert_qq_segments
from .expression_rag.retrieval import build_prompt, embedding_text, select_examples
from .expression_rag.store import ExpressionStore


PLUGIN_ID = "alan.expression-rag"


def read_seed_entries(plugin_dir: Path) -> List[dict]:
    """Read only an explicitly supplied local seed; public installs start empty."""
    seed_file = plugin_dir / "assets" / "seed_expressions.json"
    if not seed_file.is_file():
        return []
    seed = json.loads(seed_file.read_text(encoding="utf-8"))
    if not isinstance(seed, list):
        raise ValueError("初始表达文件必须是 JSON 数组")
    return seed


def _display_example(entry: dict) -> dict:
    """Bound WebUI payloads without changing the stored or retrieved example."""
    contents = entry.get("content_list", [])
    truncated = len(contents) > 4 or any(len(text) > 300 for text in contents)
    result = {
        "id": entry["id"], "situation": entry["situation"], "style": entry["style"],
        "content_list": [text[:300] + ("…" if len(text) > 300 else "") for text in contents[:4]],
        "source": entry.get("source", ""), "display_truncated": truncated,
    }
    if "score" in entry:
        result["score"] = entry["score"]
    return result


def _display_result(result: dict) -> dict:
    return {**result, "items": [_display_example(entry) for entry in result.get("items", [])]}


def _message_rows(payload: Any) -> List[dict]:
    """Unwrap only ordinary Host message responses, never any memory data."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if payload.get("success") is False:
            raise RuntimeError("无法读取当前会话的普通聊天消息")
        for key in ("messages", "data", "message"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
            if isinstance(value, dict):
                return [value]
        if any(key in payload for key in ("processed_plain_text", "plain_text", "text", "message_id")):
            return [payload]
    return []


def _message_text(row: dict) -> str:
    for key in ("processed_plain_text", "plain_text", "text", "display_message"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_query(target: Any, recent: Any, max_chars: int) -> str:
    """Emphasize the actual reply target over unrelated nearby group messages."""
    targets = _message_rows(target)
    target_text = _message_text(targets[0]) if targets else ""
    target_id = str(targets[0].get("message_id", "")) if targets else ""
    context = []
    for row in _message_rows(recent):
        text = _message_text(row)
        if not text or text == target_text or (target_id and str(row.get("message_id", "")) == target_id):
            continue
        context.append(text[:350])
    if target_text:
        target_block = "当前要回复的话：" + target_text
        remaining = max_chars - len(target_block)
        nearby = "\n附近对话：" + "\n".join(context)
        return (target_block + (nearby[-remaining:] if remaining > 12 and context else ""))[:max_chars]
    return "\n".join(context)[-max_chars:].strip()


class ExpressionRAGPlugin(MaiBotPlugin):
    config_model = ExpressionRAGConfig
    config_reload_subscriptions = ("model",)

    def __init__(self) -> None:
        super().__init__()
        self._store: ExpressionStore | None = None
        self._build_task: asyncio.Task | None = None
        self._preview_task: asyncio.Task | None = None
        self._job: dict = {"state": "idle", "processed": 0, "total": 0, "error": ""}
        self._preview: dict = {"state": "idle", "items": [], "error": ""}
        self._last_retrieval: dict = {"state": "waiting", "items": [], "error": ""}
        self._cache: OrderedDict = OrderedDict()
        self._host_epoch = 0
        self._initial_count = 0

    @property
    def store(self) -> ExpressionStore:
        if self._store is None:
            raise RuntimeError("表达库尚未初始化")
        return self._store

    async def on_load(self) -> None:
        data_dir = Path(self.ctx.paths.data_dir)
        self._store = await asyncio.to_thread(ExpressionStore, data_dir / "expressions.sqlite3")
        # An optional, explicit local seed is the sole initial corpus source.
        seed = await asyncio.to_thread(read_seed_entries, Path(__file__).parent)
        self._initial_count = len(seed)
        await asyncio.to_thread(self.store.initialize, seed)
        self._host_epoch = int(await asyncio.to_thread(self.store.get_meta, "host_epoch", 0))
        self.ctx.logger.info("表达检索已加载：独立精选表达库，不读取或启用长期记忆；未自动请求 embedding")

    async def on_unload(self) -> None:
        tasks = [task for task in (self._build_task, self._preview_task) if task and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        del config_data, version
        self._cache.clear()
        self._last_retrieval = {"state": "waiting", "items": [], "error": "配置已更新，等待下一次检索"}
        if self._preview_task and not self._preview_task.done():
            self._preview_task.cancel()
            await asyncio.gather(self._preview_task, return_exceptions=True)
        self._preview = {"state": "idle", "items": [], "error": ""}
        if self._store is not None and scope == "model":
            # A global model change invalidates Host-mode geometry, without reading
            # the broadcast's possibly sensitive config fields.
            self._host_epoch += 1
            await asyncio.to_thread(self.store.set_meta, "host_epoch", self._host_epoch)
        if self._build_task and not self._build_task.done():
            self._build_task.cancel()

    def _signature(self) -> str:
        return settings_signature(self.config.embedding, self._host_epoch)

    def _client(self, settings: Any = None) -> EmbeddingClient:
        return EmbeddingClient(settings or self.config.embedding.model_copy(deep=True), self.ctx)

    async def _spec(self, signature: str) -> dict | None:
        return await asyncio.to_thread(self.store.get_meta, "index:" + signature)

    async def _retrieve(self, query: str) -> List[dict]:
        signature = self._signature()
        spec = await self._spec(signature)
        if not spec or not spec.get("complete"):
            raise EmbeddingError("当前模型尚未完成索引，请先在表达库页面建立或更新索引")
        if await asyncio.to_thread(self.store.get_meta, "invalid:" + signature, False):
            raise EmbeddingError("模型空间已变化，请修改索引版本并重新建立索引")
        cfg = self.config.retrieval
        cache_payload = [signature, query, cfg.top_k, cfg.fetch_k, cfg.min_similarity, cfg.mmr_lambda]
        cache_key = hashlib.sha256(json.dumps(cache_payload, ensure_ascii=False).encode("utf-8")).hexdigest()
        cached = self._cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 120:
            self._cache.move_to_end(cache_key)
            return cached[1]
        batch = await self._client().embed([query])
        if batch.model_name != spec["model_name"] or len(batch.vectors[0]) != spec["dimension"]:
            await asyncio.to_thread(self.store.set_meta, "invalid:" + signature, True)
            raise EmbeddingError("查询模型或维度与索引不同，已停止使用旧向量；请修改索引版本并重新建立")
        if signature != self._signature():
            raise EmbeddingError("检索期间模型配置已变化，请重试")
        entries, vectors = await asyncio.gather(
            asyncio.to_thread(self.store.entries), asyncio.to_thread(self.store.vectors, spec["namespace"])
        )
        hits = await asyncio.to_thread(
            select_examples, entries, vectors, batch.vectors[0],
            k=cfg.top_k, fetch_k=max(cfg.top_k, cfg.fetch_k),
            min_similarity=cfg.min_similarity, mmr_lambda=cfg.mmr_lambda,
        )
        self._cache[cache_key] = (time.monotonic(), hits)
        while len(self._cache) > 128:
            self._cache.popitem(last=False)
        return hits

    async def _build_index(self, signature: str, settings: Any) -> None:
        self._job = {"state": "running", "processed": 0, "total": 0, "error": ""}
        try:
            entries = await asyncio.to_thread(self.store.entries)
            spec = await self._spec(signature)
            vectors = await asyncio.to_thread(self.store.vectors, spec["namespace"]) if spec else {}
            pending = [entry for entry in entries if entry["id"] not in vectors]
            self._job.update(total=len(entries), processed=len(entries) - len(pending))
            client = self._client(settings)
            for offset in range(0, len(pending), settings.batch_size):
                if signature != self._signature():
                    raise EmbeddingError("配置已改变，本次建立索引已停止；旧索引保留")
                group = pending[offset:offset + settings.batch_size]
                batch = await client.embed([embedding_text(entry) for entry in group])
                dimension = len(batch.vectors[0])
                if spec is None:
                    spec = {
                        "namespace": vector_namespace(signature, batch.model_name, dimension),
                        "model_name": batch.model_name, "dimension": dimension, "complete": False,
                    }
                elif spec["model_name"] != batch.model_name or spec["dimension"] != dimension:
                    await asyncio.to_thread(self.store.set_meta, "invalid:" + signature, True)
                    raise EmbeddingError("模型或维度发生变化，未混写向量；请修改索引版本后重新建立")
                if signature != self._signature():
                    raise EmbeddingError("请求期间配置已改变，本次结果未写入索引")
                pairs = [(entry["id"], vector) for entry, vector in zip(group, batch.vectors)]
                await asyncio.to_thread(self.store.put_vectors, spec["namespace"], pairs)
                await asyncio.to_thread(self.store.set_meta, "index:" + signature, spec)
                self._job["processed"] += len(group)
            if spec is None:
                raise EmbeddingError("表达库为空，不能建立索引")
            spec["complete"] = True
            await asyncio.to_thread(self.store.set_meta, "index:" + signature, spec)
            await asyncio.to_thread(self.store.set_meta, "invalid:" + signature, False)
            self._cache.clear()
            self._job["state"] = "completed"
        except asyncio.CancelledError:
            self._job.update(state="cancelled", error="配置变更或插件卸载，任务已停止；已完成的向量保留")
        except Exception as exc:
            error = self._safe_error(exc)
            self._job.update(state="failed", error=error)
            self.ctx.logger.warning("表达索引未完成：%s", error)

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if isinstance(exc, (EmbeddingError, ValueError)):
            return str(exc)[:500]
        return f"操作失败（{type(exc).__name__}）；请检查插件与服务配置"

    @API("webui_status", description="表达库、模型和索引状态；不返回密钥或聊天记录", version="1")
    async def webui_status(self) -> dict:
        entries = await asyncio.to_thread(self.store.entries)
        error, spec = "", None
        try:
            self._client().validate_configuration()
            signature = self._signature()
            spec = await self._spec(signature)
            status = await asyncio.to_thread(self.store.index_status, spec["namespace"] if spec else "unbuilt")
            invalid = await asyncio.to_thread(self.store.get_meta, "invalid:" + signature, False)
        except (EmbeddingError, ValueError) as exc:
            error = self._safe_error(exc)
            status, invalid = {"total": len(entries), "indexed": 0, "missing": len(entries)}, False
        return {
            "total": len(entries), "initial_count": self._initial_count, "indexed": status["indexed"], "missing": status["missing"],
            "ready": bool(spec and spec.get("complete") and not invalid and not error),
            "backend": self.config.embedding.backend,
            "selected_model": self.config.embedding.model if self.config.embedding.backend == "openai" else self.config.embedding.host_task,
            "model_name": spec["model_name"] if spec else "", "dimension": spec["dimension"] if spec else 0,
            "namespace": spec["namespace"][:16] if spec else "", "configuration_error": error,
            "invalid": invalid, "job": dict(self._job), "preview": _display_result(self._preview),
            "last_retrieval": _display_result(self._last_retrieval), "retrieval_enabled": self.config.retrieval.enabled,
            "source_scope": "仅使用独立表达库中明确导入的表达；不读取或启用长期记忆库",
        }

    @API("webui_entries", description="浏览独立表达库", version="1")
    async def webui_entries(self, search: str = "", offset: int = 0) -> dict:
        if not isinstance(search, str) or len(search) > 200 or type(offset) is not int or offset < 0:
            raise ValueError("检索词或页码不正确")
        entries = await asyncio.to_thread(self.store.entries)
        word = search.casefold().strip()
        rows = [entry for entry in entries if not word or word in json.dumps(entry, ensure_ascii=False).casefold()]
        return {"items": [_display_example(entry) for entry in rows[offset:offset + 12]], "total": len(rows), "offset": offset, "page_size": 12}

    @API("webui_build", description="明确请求后增量建立索引；后台任务，无自动重试", version="1")
    async def webui_build(self) -> dict:
        self._client().validate_configuration()
        signature = self._signature()
        if await asyncio.to_thread(self.store.get_meta, "invalid:" + signature, False):
            raise ValueError("当前模型空间已变化，请在插件配置中修改索引版本后重新建立")
        if self._build_task and not self._build_task.done():
            return {"state": "running", "message": "已有索引任务，未重复发起请求"}
        settings = self.config.embedding.model_copy(deep=True)
        self._job = {"state": "queued", "processed": 0, "total": 0, "error": ""}
        self._build_task = asyncio.create_task(self._build_index(signature, settings))
        return {"state": "queued", "message": "已受理，请查看进度；重复点击不会创建并行索引任务"}

    @API("webui_preview", description="检索预览；只调用 embedding，不生成或发送 QQ 消息", version="1")
    async def webui_preview(self, query: str) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("请输入1到2000字的聊天内容")
        if self._preview_task and not self._preview_task.done():
            raise ValueError("已有预览请求，请等待结果")
        self._preview = {"state": "running", "items": [], "error": ""}

        async def run() -> None:
            try:
                async with asyncio.timeout(self.config.embedding.timeout_seconds + 1):
                    hits = await self._retrieve(query.strip())
                self._preview = {"state": "completed", "items": hits, "prompt": build_prompt(hits, self.config.retrieval.prompt_max_chars), "error": ""}
            except asyncio.CancelledError:
                self._preview = {"state": "cancelled", "items": [], "error": "插件卸载，预览已停止"}
            except Exception as exc:
                self._preview = {"state": "failed", "items": [], "error": self._safe_error(exc)}

        self._preview_task = asyncio.create_task(run())
        return {"state": "running"}

    @API("webui_models", description="按用户请求读取模型列表，不自动选择模型", version="1")
    async def webui_models(self) -> dict:
        return {"items": await self._client().list_models(), "backend": self.config.embedding.backend}

    @API("webui_import", description="明确追加精选表达，不读取其它目录或删除旧条目", version="1")
    async def webui_import(self, entries_json: str) -> dict:
        if not isinstance(entries_json, str) or len(entries_json.encode("utf-8")) > 48000:
            raise ValueError("每批导入 JSON 不超过48KB，请分批导入")
        if self._build_task and not self._build_task.done():
            raise ValueError("请先等待当前索引任务完成再导入")
        try:
            items = json.loads(entries_json)
        except (ValueError, TypeError):
            raise ValueError("导入内容不是有效的 JSON") from None
        if not isinstance(items, list) or not 1 <= len(items) <= 200:
            raise ValueError("请提供包含1到200条精选表达的 JSON 数组")
        result = await asyncio.to_thread(self.store.import_entries, items, source="WebUI导入")
        self._cache.clear()
        return result

    @HookHandler("maisaka.replyer.before_request", mode=HookMode.BLOCKING, order=HookOrder.LATE, timeout_ms=7000)
    async def inject_examples(self, **kwargs: Any) -> dict:
        if not self.config.plugin.enabled or not self.config.retrieval.enabled:
            return {"action": "continue"}
        session_id = str(kwargs.get("session_id") or "")
        if not session_id:
            return {"action": "continue"}
        try:
            self._client().validate_configuration()
        except EmbeddingError:
            self._last_retrieval = {"state": "waiting", "items": [], "error": "请先选择模型并建立索引"}
            return {"action": "continue"}
        try:
            # Skip unprepared indexes before querying messages or making any API call.
            signature = self._signature()
            spec = await self._spec(signature)
            if not spec or not spec.get("complete"):
                self._last_retrieval = {"state": "waiting", "items": [], "error": "请先选择模型并建立索引"}
                return {"action": "continue"}
            async with asyncio.timeout(self.config.retrieval.reply_timeout_seconds):
                target_id = str(kwargs.get("reply_message_id") or "")
                target = await self.ctx.message.get_by_id(target_id, chat_id=session_id) if target_id else None
                recent = await self.ctx.message.get_recent(session_id, limit=self.config.retrieval.context_messages)
                query = build_query(target, recent, self.config.retrieval.query_max_chars)
                if not query:
                    self._last_retrieval = {"state": "no_context", "items": [], "error": "没有可用于检索的当前聊天文本"}
                    return {"action": "continue"}
                hits = await self._retrieve(query)
                block = build_prompt(hits, self.config.retrieval.prompt_max_chars)
            self._last_retrieval = {"state": "matched" if hits else "no_match", "items": hits, "error": ""}
            if not block:
                return {"action": "continue"}
            modified = dict(kwargs)
            modified["extra_prompt"] = (str(kwargs.get("extra_prompt") or "") + "\n\n" + block).strip()
            return {"action": "continue", "modified_kwargs": modified}
        except TimeoutError:
            error = "表达检索超过本轮等待上限，已继续正常聊天；没有随机替代向量命中"
        except Exception as exc:
            error = self._safe_error(exc)
        self._last_retrieval = {"state": "failed", "items": [], "error": error}
        self.ctx.logger.warning("表达检索未注入：%s", error)
        return {"action": "continue"}

    @HookHandler("send_service.before_send", mode=HookMode.BLOCKING, order=HookOrder.LATE, timeout_ms=1000)
    async def convert_faces(self, **kwargs: Any) -> dict:
        if not self.config.plugin.enabled or not self.config.qq_faces.enabled:
            return {"action": "continue"}
        message = kwargs.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("raw_message"), list):
            return {"action": "continue"}
        raw = convert_qq_segments(message["raw_message"], str(message.get("platform", "")))
        if raw == message["raw_message"]:
            return {"action": "continue"}
        return {"action": "continue", "modified_kwargs": {**kwargs, "message": {**message, "raw_message": raw}}}


def create_plugin() -> ExpressionRAGPlugin:
    return ExpressionRAGPlugin()
