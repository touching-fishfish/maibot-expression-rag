"""Explicit embedding backends with no automatic retries or provider switching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
from urllib.parse import urlsplit, urlunsplit

import asyncio
import hashlib
import json
import math

import httpx


class EmbeddingError(RuntimeError):
    """A user-actionable error that never includes provider response bodies or keys."""


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: List[List[float]]
    model_name: str


def api_root(value: str) -> str:
    raw = value.strip().rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise EmbeddingError("请填写有效的 embedding API 根地址")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise EmbeddingError("API 地址不能包含用户名、密码、查询参数或片段")
    path = parsed.path.rstrip("/")
    if path.endswith("/embeddings") or path.endswith("/models"):
        path = path.rsplit("/", 1)[0]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def settings_signature(settings: Any, host_epoch: int = 0) -> str:
    """Keys do not affect vector geometry, and are never written into index metadata."""
    if settings.backend == "host":
        payload = {"backend": "host", "task": settings.host_task, "revision": settings.revision, "epoch": host_epoch}
    else:
        payload = {
            "backend": "openai", "base_url": api_root(settings.base_url), "model": settings.model.strip(),
            "dimensions": settings.dimensions, "revision": settings.revision,
        }
    payload["text_format"] = "situation-style-v1"
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def vector_namespace(signature: str, model_name: str, dimension: int) -> str:
    payload = f"{signature}\n{model_name}\n{dimension}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_vectors(raw: Any, expected_count: int) -> List[List[float]]:
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise EmbeddingError("embedding 返回的向量条数与输入不一致")
    vectors = []
    dimension = None
    for values in raw:
        if not isinstance(values, list) or not 1 <= len(values) <= 32768:
            raise EmbeddingError("embedding 返回了空向量或不支持的维度")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise EmbeddingError("embedding 返回了非数值或非有限值")
        vector = [float(v) for v in values]
        if not math.isfinite(math.hypot(*vector)) or math.hypot(*vector) == 0:
            raise EmbeddingError("embedding 返回了无效的零向量")
        if dimension is not None and len(vector) != dimension:
            raise EmbeddingError("同批 embedding 的维度不一致")
        dimension = len(vector)
        vectors.append(vector)
    return vectors


class EmbeddingClient:
    def __init__(self, settings: Any, ctx: Any, *, transport: Any = None) -> None:
        self.settings = settings
        self.ctx = ctx
        self.transport = transport

    def validate_configuration(self) -> None:
        if self.settings.backend == "openai":
            api_root(self.settings.base_url)
            if not self.settings.model.strip():
                raise EmbeddingError("请先在插件配置中选择 embedding 模型")
        elif not self.settings.host_task.strip():
            raise EmbeddingError("请填写宿主 embedding 任务名")

    async def _post(self, endpoint: str, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key.strip():
            headers["Authorization"] = f"Bearer {self.settings.api_key.strip()}"
        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds, transport=self.transport) as client:
                response = await client.post(api_root(self.settings.base_url) + endpoint, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(f"embedding 服务返回 HTTP {exc.response.status_code}，请核对地址、模型与密钥") from None
        except (httpx.HTTPError, ValueError):
            raise EmbeddingError("embedding 服务连接、超时或 JSON 格式错误；未自动重试") from None
        if not isinstance(result, dict):
            raise EmbeddingError("embedding 服务返回的 JSON 不是对象")
        return result

    async def embed(self, texts: List[str]) -> EmbeddingBatch:
        try:
            async with asyncio.timeout(self.settings.timeout_seconds):
                return await self._embed(texts)
        except TimeoutError:
            raise EmbeddingError("embedding 请求超过配置的等待上限；未自动重试") from None

    async def _embed(self, texts: List[str]) -> EmbeddingBatch:
        self.validate_configuration()
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("embedding 输入不能为空")
        if self.settings.backend == "host":
            result = await self.ctx.llm.embed(texts=texts, task_name=self.settings.host_task, max_concurrent=2)
            if not isinstance(result, dict) or result.get("success") is not True:
                raise EmbeddingError("宿主 embedding 调用失败；请在模型配置中为 embedding 任务选择一个可用模型")
            rows = result.get("results")
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise EmbeddingError("宿主 embedding 返回格式不正确")
            names = {str(row.get("model_name") or "") for row in rows}
            if len(names) != 1 or "" in names:
                raise EmbeddingError("宿主任务返回了多个模型或缺少模型标识；请只选择一个 embedding 模型")
            return EmbeddingBatch(validate_vectors([row.get("embedding") for row in rows], len(texts)), names.pop())

        body = {"model": self.settings.model.strip(), "input": texts, "encoding_format": "float"}
        if self.settings.dimensions:
            body["dimensions"] = self.settings.dimensions
        result = await self._post("/embeddings", body)
        rows = result.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts) or any(not isinstance(row, dict) for row in rows):
            raise EmbeddingError("embedding 服务的 data 条数或结构不正确")
        indices = [row.get("index") for row in rows]
        if any(type(index) is not int for index in indices) or sorted(indices) != list(range(len(texts))):
            raise EmbeddingError("embedding 返回缺失、重复或非法的输入编号")
        rows = sorted(rows, key=lambda row: row["index"])
        vectors = validate_vectors([row.get("embedding") for row in rows], len(texts))
        if self.settings.dimensions and any(len(vector) != self.settings.dimensions for vector in vectors):
            raise EmbeddingError("服务商返回的向量维度与指定维度不同")
        return EmbeddingBatch(vectors, str(result.get("model") or self.settings.model.strip()))

    async def list_models(self) -> List[str]:
        if self.settings.backend == "host":
            try:
                async with asyncio.timeout(6):
                    return list(await self.ctx.llm.get_available_models())
            except TimeoutError:
                raise EmbeddingError("读取宿主模型任务列表超时") from None
        headers = {}
        if self.settings.api_key.strip():
            headers["Authorization"] = f"Bearer {self.settings.api_key.strip()}"
        try:
            async with httpx.AsyncClient(timeout=6, transport=self.transport) as client:
                response = await client.get(api_root(self.settings.base_url) + "/models", headers=headers)
                response.raise_for_status()
                result = response.json()
        except (httpx.HTTPError, ValueError):
            raise EmbeddingError("读取模型列表失败；可直接在标准插件配置页填写服务商的 embedding 模型标识") from None
        rows = result.get("data") if isinstance(result, dict) else None
        if not isinstance(rows, list):
            raise EmbeddingError("服务商未返回标准模型列表；请手动填写 embedding 模型标识")
        return sorted({str(row["id"]) for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)})[:500]
