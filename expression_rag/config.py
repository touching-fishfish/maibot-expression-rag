"""Typed settings rendered by MaiBot's standard plugin configuration page."""

from __future__ import annotations

from typing import Literal

from maibot_sdk import Field, PluginConfigBase


class PluginSection(PluginConfigBase):
    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="启用表达检索插件；未配置模型或未建立索引时不发起自动请求")
    config_version: str = Field(default="1.0.0", description="配置版本", json_schema_extra={"hidden": True})


class EmbeddingSection(PluginConfigBase):
    __ui_label__ = "Embedding 模型"
    __ui_icon__ = "settings"
    __ui_order__ = 1

    backend: Literal["openai", "host"] = Field(
        default="openai", description="openai：独立选择兼容服务商和模型；host：使用主程序 embedding 任务"
    )
    base_url: str = Field(
        default="", max_length=500,
        description="兼容 API 的根地址，包含服务商要求的 /v1 或 /api/v3；插件不会擅自追加版本段",
    )
    model: str = Field(default="", max_length=200, description="由你选择的具体 embedding 模型标识；不要填写聊天模型")
    api_key: str = Field(
        default="", max_length=1000, description="该 embedding 服务商的密钥；只存插件本地配置",
        json_schema_extra={"x-widget": "password"},
    )
    dimensions: int = Field(default=0, ge=0, le=32768, description="0 表示由模型返回默认维度；仅服务商支持时填写指定维度")
    host_task: str = Field(default="embedding", min_length=1, max_length=100, description="host 模式的任务名；请在主程序中为该任务只选择一个模型")
    revision: str = Field(
        default="1", min_length=1, max_length=100,
        description="索引版本。host 模式在离线更换了同名模型或服务商时，请修改此值后重新建立索引",
    )
    timeout_seconds: float = Field(default=20, ge=1, le=90, description="建立索引时每批请求最长等待秒数；回复时另有总等待上限")
    batch_size: int = Field(default=8, ge=1, le=32, description="每批索引的条目数；host 模式最多并发两个嵌入请求")


class RetrievalSection(PluginConfigBase):
    __ui_label__ = "表达检索"
    __ui_icon__ = "database"
    __ui_order__ = 2

    enabled: bool = Field(default=True, description="决定回复后，检索相近表达并加入本轮回复要求")
    top_k: int = Field(default=3, ge=1, le=8, description="最终提供给回复模型的表达条数")
    fetch_k: int = Field(default=12, ge=1, le=64, description="先取相近候选，再从中选出较少重复的例子")
    min_similarity: float = Field(default=0.25, ge=-1, le=1, description="余弦相似度低于此值的表达不注入；不同模型需要实际调试")
    mmr_lambda: float = Field(default=0.75, ge=0, le=1, description="相关性权重；越高越重视贴题，越低越重视表达多样性")
    context_messages: int = Field(default=4, ge=1, le=10, description="只读取当前会话最近几条普通聊天消息，不查询任何记忆库")
    query_max_chars: int = Field(default=1200, ge=100, le=3000, description="发送给 embedding 的当前聊天文本长度上限")
    prompt_max_chars: int = Field(default=2400, ge=600, le=6000, description="注入的表达示例总长度上限")
    reply_timeout_seconds: float = Field(default=5, ge=1, le=6, description="本轮检索总等待上限；超时会明确记录并继续正常聊天")


class FaceSection(PluginConfigBase):
    __ui_label__ = "QQ 表情"
    __ui_icon__ = "message-circle"
    __ui_order__ = 3

    enabled: bool = Field(default=True, description="将精选语料中的五种已知表情转为原生 QQ 表情；只处理 QQ 出站文本，不修改语料")


class ExpressionRAGConfig(PluginConfigBase):
    plugin: PluginSection = Field(default_factory=PluginSection)
    embedding: EmbeddingSection = Field(default_factory=EmbeddingSection)
    retrieval: RetrievalSection = Field(default_factory=RetrievalSection)
    qq_faces: FaceSection = Field(default_factory=FaceSection)
