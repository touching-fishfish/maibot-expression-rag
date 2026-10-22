# 精选表达检索 · alan.expression-rag

在 MaiBot 已经决定说话之后，使用当前会话的聊天文本检索相近的表达例子，再让回复模型参考其口吻与回应方法。要求 MaiBot ≥ 1.3.12（当前按 1.3.12 验证）、SDK ≥ 2.12、WebUI 页面协议 2。仓库公开分发：随包 68 条脱敏示例语料；不含密钥与本机配置。

## 安装

插件不依赖固定目录或文件夹名称；运行数据只使用宿主提供的插件数据目录，无需自建向量数据库服务。

两种装法任选其一：

1. **从 Git 仓库克隆（推荐）**：MaiBot WebUI → 插件管理 → 安装插件 → 从仓库克隆，填 `https://github.com/touching-fishfish/maibot-expression-rag`。
2. **下载 zip 手动解压**：在仓库页 Download ZIP，把解压出的整个文件夹放进 MaiBot 的 `plugins/` 目录（文件夹名不限），重启 MaiBot 加载。

随包附带 **68 条脱敏示例语料**（`assets/seed_expressions.json`，仅“场合/招式”，无姓名、地名、号码或聊天原文），首次建立索引时自动初始化；之后仍可随时在 WebUI 追加你自己的语料。原始聊天记录、`config.toml`、API 密钥与本机向量数据均不进入仓库。

迁移时备份并复制插件专属数据目录即可，模型配置与密钥在目标机器单独填写；插件不会自动删除任何数据。

## 数据边界

- 随包 68 条示例语料已脱敏：只保留场景与表达方式，不含姓名、地点、号码或聊天原文；不需要可整体删除后从空库开始。
- **不读取、不查询、不导入宿主的封存记忆库**；不申请 memory、person、knowledge 或主数据库能力，也不打开 MaiBot.db / a-memorix。
- embedding 的索引文本只包含“聊天场景＋表达方式”；参考句保留为检索结果，不把其中的个人经历当作事实。
- 当前聊天仅通过宿主普通消息接口读取，限定本次回复的真实会话 ID；目标消息优先，上下文长度有上限。查询缓存只在内存保留，不存聊天文本到插件数据库。
- 新表达只能由你在页面中明确追加导入；不自动学习、扫描目录、删除语料或清理旧索引。原始微信文件保持原样。

## 第一次使用

1. 在 MaiBot WebUI 的插件管理中找到“精选表达检索”，启用插件。
2. 打开它的标准配置页，选择 embedding 方式并保存。
   - `openai`：填写你选择的兼容服务商 `base_url`、embedding `model` 和 `api_key`。地址保留服务商要求的 `/v1` 或 `/api/v3`，插件不会自动拼接版本号。密钥也可留空用于本机无认证服务。示例：硅基流动 `https://api.siliconflow.cn/v1` + `BAAI/bge-m3`（1024 维，免费额度）或 `Qwen/Qwen3-Embedding-8B`。
   - `host`：复用主程序 `embedding` 任务。请在主程序模型配置中为该任务选择**唯一一个** embedding 模型，插件不替你选择模型。此模式不需要在插件再填密钥。
3. 打开顶部“表达检索”工作区，点击“建立 / 更新索引”，等待已索引 68 条（随包语料）或与你导入的总条数一致。加载插件本身不会调用 embedding；只有你建立索引、做预览或索引就绪后的真实回复会调用。
4. 在“检索预览”输入一段聊天，观察匹配场景、参考句、真实余弦分数和将注入的提示文本。预览不会生成或发送 QQ 消息。
5. 实际聊天时，插件在 `maisaka.replyer.before_request` Hook 中加入命中的例子；保留人设、其它插件的附加提示和原回复流程。无合适结果时允许不注入。

默认最终选择3条、候选12条、最低余弦相似度0.25、MMR相关性权重0.75。相似度阈值取决于模型，需要用你的真实聊天预览调整，不代表验收成绩。

本机主程序还会产生自己的原生表达建议；插件不修改它的配置或源代码，而在本轮附加提示中明确优先参考语义命中的示例，不生搬随机表达。

## 更换模型与追加语料

服务地址、模型、指定维度或索引版本改变，会使用新的索引空间，不混用旧向量。运行期间宿主模型配置变更也会使 host 索引失效。若在机器人关闭时替换了同名模型或服务商，请修改插件的 `revision` 再建立索引。返回的实际模型名和维度还会在查询时校验。

建立索引只编码缺失条目；已完成批次会保存，失败后再次点击可以继续。没有自动重试、自动换服务商或假装成功的随机检索。配置变更会停止当前任务，已经提交给服务商的请求可能仍会计费。

追加导入接受 JSON 数组，每批1–200条、UTF-8最多48,000字节；重复内容跳过，原有条目保留。示例：

```json
[
  {
    "situation": "对方分享好消息",
    "style": "轻轻肯定一句，不夸张吹捧",
    "content_list": ["不错啊", "有实力"]
  }
]
```

导入后再次更新索引，才能检索新增部分。页面没有删除按钮。

## QQ 表情

保留语料中的表情，在 QQ 发送前将 `[捂脸]`、`[可怜]`、`[破涕为笑]`、`[大哭]`、`😋` 转为适配器可编码的原生 face 段。未知表情、非文本消息和其它平台保持原样。转换幂等，可以在标准配置页独立关闭。

这五种映射由本插件独立实现，不依赖其他插件；face ID 已按 SnowLuma 适配器核对，其中“破涕为笑”使用 QQ 笑哭、😋 使用适配器既有饥饿映射，外观与微信略有差异。

## 存储与验证

持久数据仅位于宿主提供的 `data/plugins/alan.expression-rag/expressions.sqlite3`。密钥由宿主管理，保存在本插件 `config.toml`，不进入索引元数据或状态 API。

离线测试：在本插件目录执行 `python -m unittest discover -s tests -v`（可用 MaiBot 运行环境的 Python）。测试使用独立测试数据库和模拟 embedding，不调用付费接口，不发送 QQ 消息。

代码独立实现，参考以下一手项目的设计，没有把它们作为运行依赖：

- [LangChain 语义示例选择与 MMR](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/example_selectors/semantic_similarity.py)
- [LangChain 向量相似度与 MMR](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores/utils.py)
- [LlamaIndex 内容与转换配置的缓存指纹](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/ingestion/pipeline.py)
- [MaiBot 官方插件开发指南](https://github.com/Mai-with-u/maibot-plugin-sdk/blob/main/docs/guide.md)，实际调用以本机 SDK 2.12 为准。

仓库：<https://github.com/touching-fishfish/maibot-expression-rag>。代码与随包示例语料均按 MIT 许可分发；原始微信聊天记录不随仓库分发，也不授权公开再分发。
