"""Offline SDK integration and provider-contract checks; no external requests."""

from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase

import asyncio
import importlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import unittest
import uuid

import httpx


ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE = "expression_rag_plugin_under_test"
_SPEC = importlib.util.spec_from_file_location(TEST_PACKAGE, ROOT / "plugin.py", submodule_search_locations=[str(ROOT)])
assert _SPEC is not None and _SPEC.loader is not None
_PLUGIN = importlib.util.module_from_spec(_SPEC)
sys.modules[TEST_PACKAGE] = _PLUGIN
_SPEC.loader.exec_module(_PLUGIN)
_CONFIG = importlib.import_module(TEST_PACKAGE + ".expression_rag.config")
_EMBEDDING = importlib.import_module(TEST_PACKAGE + ".expression_rag.embedding")
EmbeddingSection = _CONFIG.EmbeddingSection
EmbeddingClient, EmbeddingError = _EMBEDDING.EmbeddingClient, _EMBEDDING.EmbeddingError
settings_signature, validate_vectors = _EMBEDDING.settings_signature, _EMBEDDING.validate_vectors
ExpressionRAGPlugin, build_query = _PLUGIN.ExpressionRAGPlugin, _PLUGIN.build_query


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.model = "offline-test-embedding"
        self.delay = 0

    async def embed(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        return {"success": True, "results": [
            {"model_name": self.model, "embedding": [1.0, 0.25, 0.1]}
            for _ in kwargs["texts"]
        ]}


class FakeMessages:
    def __init__(self):
        self.queries = []

    async def get_by_id(self, message_id, *, chat_id):
        self.queries.append(("target", message_id, chat_id))
        return {"success": True, "message": {"message_id": message_id, "processed_plain_text": "今天有点累"}}

    async def get_recent(self, session_id, limit):
        self.queries.append(("recent", session_id, limit))
        return {"success": True, "messages": [{"message_id": "older", "processed_plain_text": "课刚上完"}]}


class PluginFlowTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.data_dir = ROOT / "tests" / ".test_artifacts" / uuid.uuid4().hex
        self.data_dir.mkdir(parents=True)
        self.llm, self.messages = FakeLLM(), FakeMessages()
        self.plugin = ExpressionRAGPlugin()
        self.plugin.set_plugin_config({"plugin": {"config_version": "1.0.0"}, "embedding": {"backend": "host"}})
        # No memory, person, knowledge or db capabilities exist in this context.
        self.plugin._set_context(SimpleNamespace(
            paths=SimpleNamespace(data_dir=self.data_dir), llm=self.llm, message=self.messages,
            logger=logging.getLogger("expression-rag-offline-test"),
        ))
        await self.plugin.on_load()

    async def asyncTearDown(self):
        await self.plugin.on_unload()

    async def build(self):
        await self.plugin.webui_build()
        await self.plugin._build_task
        self.assertEqual(self.plugin._job["state"], "completed", self.plugin._job)

    async def test_load_seeds_exact_68_without_network_or_memory(self):
        self.assertEqual(len(self.plugin.store.entries()), 68)
        self.assertEqual(self.llm.calls, [])
        status = await self.plugin.webui_status()
        self.assertEqual((status["indexed"], status["missing"]), (0, 68))
        self.assertFalse(status["ready"])
        self.assertEqual(status["configuration_error"], "")
        self.assertNotIn("api_key", json.dumps(status))

    async def test_frozen_seed_matches_only_both_curated_sources(self):
        corpus = ROOT.parents[3] / "corpus" / "curated" if len(ROOT.parents) > 3 else None
        expected = []
        for path in sorted(corpus.glob("entry-*.json")) if corpus is not None else []:
            expected.extend(json.loads(path.read_text(encoding="utf-8-sig")))
        rows = self.plugin.store.entries()
        actual = [{key: row[key] for key in ("situation", "style", "content_list")} for row in rows]
        self.assertEqual(len(actual), 68)
        if corpus is not None and corpus.exists():
            self.assertEqual(len(expected), 68)
            self.assertEqual(actual, expected)

    async def test_unbuilt_index_never_queries_chat_or_embedding(self):
        result = await self.plugin.inject_examples(session_id="session-A", reply_message_id="m1")
        self.assertEqual(result, {"action": "continue"})
        self.assertEqual(self.messages.queries, [])
        self.assertEqual(self.llm.calls, [])

    async def test_build_incremental_and_preserves_prior_vectors(self):
        await self.build()
        status = await self.plugin.webui_status()
        self.assertEqual((status["total"], status["indexed"], status["missing"]), (68, 68, 0))
        first_calls = len(self.llm.calls)
        await self.build()
        self.assertEqual(len(self.llm.calls), first_calls)
        item = {"situation": "新增测试场景", "style": "轻轻回应", "content_list": ["行呀[捂脸]"]}
        result = await self.plugin.webui_import(json.dumps([item], ensure_ascii=False))
        self.assertEqual(result["added"], 1)
        status = await self.plugin.webui_status()
        self.assertEqual((status["indexed"], status["missing"]), (68, 1))
        await self.build()
        self.assertEqual(len(self.llm.calls), first_calls + 1)
        self.assertEqual((await self.plugin.webui_status())["indexed"], 69)

    async def test_reply_hook_preserves_other_plugins_and_scope_and_reuses_retry(self):
        await self.build()
        original = {"session_id": "session-A", "reply_message_id": "m1", "extra_prompt": "已有其它插件提示", "attempt": 1}
        result = await self.plugin.inject_examples(**original)
        updated = result["modified_kwargs"]
        self.assertTrue(updated["extra_prompt"].startswith(original["extra_prompt"]))
        self.assertIn("不要把示例认作自己的经历", updated["extra_prompt"])
        self.assertEqual(original["extra_prompt"], "已有其它插件提示")
        self.assertEqual(self.messages.queries[0], ("target", "m1", "session-A"))
        self.assertEqual(self.messages.queries[1][1], "session-A")
        calls = len(self.llm.calls)
        await self.plugin.inject_examples(**dict(original, attempt=2))
        self.assertEqual(len(self.llm.calls), calls)
        self.assertEqual(self.plugin._last_retrieval["state"], "matched")

    async def test_different_runtime_model_rejects_old_vectors(self):
        await self.build()
        signature = self.plugin._signature()
        spec = await self.plugin._spec(signature)
        self.llm.model = "another-embedding-model"
        with self.assertRaises(EmbeddingError):
            await self.plugin._retrieve("a new query")
        self.assertEqual(len(self.plugin.store.vectors(spec["namespace"])), 68)
        self.assertFalse((await self.plugin.webui_status())["ready"])

    async def test_global_model_change_invalidates_only_host_space(self):
        await self.build()
        previous_signature = self.plugin._signature()
        previous = await self.plugin._spec(previous_signature)
        await self.plugin.on_config_update("model", {}, "new")
        self.assertNotEqual(previous_signature, self.plugin._signature())
        self.assertFalse((await self.plugin.webui_status())["ready"])
        self.assertEqual(len(self.plugin.store.vectors(previous["namespace"])), 68)

    async def test_reply_timeout_continues_without_injection(self):
        await self.build()
        cfg = self.plugin.get_plugin_config_data()
        cfg["retrieval"]["reply_timeout_seconds"] = 1
        self.plugin.set_plugin_config(cfg)
        self.llm.delay = 2
        result = await self.plugin.inject_examples(session_id="session-A", reply_message_id="slow")
        self.assertEqual(result, {"action": "continue"})
        self.assertEqual(self.plugin._last_retrieval["state"], "failed")

    async def test_face_hook_preserves_other_fields_and_original_message(self):
        raw = [{"type": "text", "data": "嗯[捂脸]"}]
        original = {"message": {"platform": "qq", "raw_message": raw, "other": "keep"}, "trace": "trace1"}
        result = await self.plugin.convert_faces(**original)
        updated = result["modified_kwargs"]
        self.assertEqual(updated["trace"], "trace1")
        self.assertEqual(updated["message"]["other"], "keep")
        self.assertEqual(raw, [{"type": "text", "data": "嗯[捂脸]"}])
        self.assertEqual(updated["message"]["raw_message"][1]["data"]["type"], "face")

    async def test_long_examples_are_summarized_below_webui_limit(self):
        for index in range(12):
            item = {"situation": str(index) + "长" * 390, "style": "式" * 1190,
                    "content_list": ["句" * 600 for _ in range(20)]}
            await self.plugin.webui_import(json.dumps([item], ensure_ascii=False))
        response = await self.plugin.webui_entries(offset=68)
        self.assertLess(len(json.dumps(response, ensure_ascii=False).encode("utf-8")), 524288)
        self.assertTrue(response["items"][0]["display_truncated"])
        self.assertEqual(len(self.plugin.store.entries()[-1]["content_list"][-1]), 600)
        self.plugin._preview = {"state": "completed", "items": self.plugin.store.entries()[-8:], "error": ""}
        self.plugin._last_retrieval = dict(self.plugin._preview)
        status = await self.plugin.webui_status()
        self.assertLess(len(json.dumps(status, ensure_ascii=False).encode("utf-8")), 524288)

    async def test_unconfigured_status_and_hook_make_no_requests(self):
        self.plugin.set_plugin_config({})
        status = await self.plugin.webui_status()
        self.assertTrue(status["configuration_error"])
        self.assertEqual(status["total"], 68)
        await self.plugin.inject_examples(session_id="session-A")
        self.assertEqual(self.messages.queries, [])
        self.assertEqual(self.llm.calls, [])


class ProviderContractTests(IsolatedAsyncioTestCase):
    def settings(self):
        return EmbeddingSection(base_url="https://embedding.invalid/api/v3", model="chosen-model", api_key="fake-test-key")

    async def test_preserves_version_path_and_orders_provider_indices(self):
        def response(request):
            self.assertEqual(request.url.path, "/api/v3/embeddings")
            self.assertEqual(json.loads(request.content)["model"], "chosen-model")
            return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]})
        client = EmbeddingClient(self.settings(), None, transport=httpx.MockTransport(response))
        result = await client.embed(["first", "second"])
        self.assertEqual(result.vectors, [[1.0, 0.0], [0.0, 1.0]])

    async def test_duplicate_input_indices_are_rejected(self):
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [1]}]}))
        with self.assertRaises(EmbeddingError):
            await EmbeddingClient(self.settings(), None, transport=transport).embed(["first", "second"])

    async def test_provider_error_does_not_echo_body_or_key(self):
        transport = httpx.MockTransport(lambda _: httpx.Response(401, text="fake-test-key private body"))
        with self.assertRaises(EmbeddingError) as caught:
            await EmbeddingClient(self.settings(), None, transport=transport).embed(["first"])
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("fake-test-key", str(caught.exception))
        self.assertNotIn("private body", str(caught.exception))

    async def test_multiple_host_models_are_rejected(self):
        class Mixed:
            async def embed(self, **kwargs):
                return {"success": True, "results": [{"model_name": "one", "embedding": [1]}, {"model_name": "two", "embedding": [1]}]}
        client = EmbeddingClient(EmbeddingSection(backend="host"), SimpleNamespace(llm=Mixed()))
        with self.assertRaises(EmbeddingError):
            await client.embed(["one", "two"])

    async def test_host_batch_respects_configured_timeout(self):
        llm = FakeLLM()
        llm.delay = 3
        client = EmbeddingClient(EmbeddingSection(backend="host", timeout_seconds=1), SimpleNamespace(llm=llm))
        with self.assertRaisesRegex(EmbeddingError, "等待上限"):
            await client.embed(["one"])


class PureBoundaryTests(TestCase):
    @unittest.skipUnless((ROOT.parents[1] / "src" / "plugin_runtime").is_dir(), "需在MaiBot宿主源码环境验证加载器")
    def test_portable_install_with_renamed_directory_and_unrelated_working_directory(self):
        workspace = ROOT / "tests" / ".test_artifacts" / uuid.uuid4().hex
        relocated = workspace / "another-instance" / "plugins" / "renamed plugin 测试"
        relocated.mkdir(parents=True)
        for name in ("plugin.py", "_manifest.json", "webui.json"):
            shutil.copy2(ROOT / name, relocated / name)
        for name in ("assets", "expression_rag", "webui"):
            shutil.copytree(ROOT / name, relocated / name, ignore=shutil.ignore_patterns("__pycache__"))
        unrelated = workspace / "unrelated-working-directory"
        unrelated.mkdir()
        script = """
from pathlib import Path
from types import SimpleNamespace
import asyncio, json, logging, sys
sys.path.insert(0, sys.argv[1])
from src.plugin_runtime.runner.plugin_loader import PluginLoader
from src.plugin_runtime.runner.manifest_validator import PluginManifest
from src.plugin_runtime.webui_schema import load_webui_extension
p = Path(sys.argv[2]); data = Path(sys.argv[3])
manifest = PluginManifest.model_validate(json.loads((p / '_manifest.json').read_text(encoding='utf-8')))
meta = PluginLoader(host_version='1.3.12')._load_single_plugin(manifest.id, p, manifest, p / 'plugin.py', for_config_request=True)
instance = meta.instance
instance.set_plugin_config(instance.build_default_config())
assert len(instance.get_webui_config_schema(plugin_id=manifest.id)['sections']) == 4
assert load_webui_extension(str(p)).schema_version == 2
instance._set_context(SimpleNamespace(paths=SimpleNamespace(data_dir=data), logger=logging.getLogger('portable-test')))
asyncio.run(instance.on_load())
assert len(instance.store.entries()) == 68
assert instance.store.path.parent.resolve() == data.resolve()
assert 'expression_rag' not in sys.modules
print('PORTABLE_RENAMED_INSTALL_OK')
"""
        result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", script,
                                 str(ROOT.parents[1]), str(relocated), str(workspace / "runtime-data")],
                                cwd=unrelated, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PORTABLE_RENAMED_INSTALL_OK", result.stdout)

    @unittest.skipUnless((ROOT.parents[1] / "src" / "plugin_runtime").is_dir(), "需在MaiBot宿主源码环境验证加载器")
    def test_actual_host_loader_builds_config_without_plugin_path_in_sys_path(self):
        app = ROOT.parents[1]
        script = """
from pathlib import Path
import json, sys
app = Path.cwd()
sys.path.insert(0, str(app))
from src.plugin_runtime.runner.plugin_loader import PluginLoader
from src.plugin_runtime.runner.manifest_validator import PluginManifest
p = app / 'plugins' / 'alan_expression-rag'
assert str(p) not in sys.path
assert 'expression_rag' not in sys.modules
manifest = PluginManifest.model_validate(json.loads((p / '_manifest.json').read_text(encoding='utf-8')))
loader = PluginLoader(host_version='1.3.12')
meta = loader._load_single_plugin(manifest.id, p, manifest, p / 'plugin.py', for_config_request=True)
assert meta is not None and meta.instance is not None
meta.instance.set_plugin_config(meta.instance.build_default_config())
schema = meta.instance.get_webui_config_schema(plugin_id=manifest.id)
assert len(schema['sections']) == 4
assert 'expression_rag' not in sys.modules
assert str(p) not in sys.path
print('HOST_LOADER_CONFIG_OK')
"""
        result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", script], cwd=app,
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("HOST_LOADER_CONFIG_OK", result.stdout)

    def test_query_prioritizes_reply_target_and_stays_bounded(self):
        target = {"success": True, "message": {"message_id": "target", "processed_plain_text": "真正要回的话"}}
        recent = {"success": True, "messages": [{"message_id": "other", "processed_plain_text": "其它群聊" * 500}]}
        query = build_query(target, recent, 100)
        self.assertTrue(query.startswith("当前要回复的话：真正要回的话"))
        self.assertLessEqual(len(query), 100)

    def test_keys_do_not_enter_geometry_fingerprint(self):
        config = EmbeddingSection(base_url="https://embedding.invalid/v1", model="chosen", api_key="one")
        old = settings_signature(config)
        self.assertEqual(old, settings_signature(config.model_copy(update={"api_key": "two"})))
        self.assertNotEqual(old, settings_signature(config.model_copy(update={"model": "other"})))

    def test_zero_nan_and_boolean_vectors_fail(self):
        for invalid in ([[0, 0]], [[float("nan")]], [[True]]):
            with self.subTest(invalid=invalid), self.assertRaises(EmbeddingError):
                validate_vectors(invalid, 1)


if __name__ == "__main__":
    unittest.main()
