from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object], text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload, ensure_ascii=False)

    def json(self) -> dict[str, object]:
        return self._payload


class FakeGenerateContentConfig:
    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class MainTests(unittest.TestCase):
    def test_fetch_latest_news_parses_articles(self) -> None:
        response = FakeResponse(
            200,
            {
                "articles": [
                    {
                        "title": "First title",
                        "description": "First description",
                        "content": "First content",
                        "url": "https://example.com/article-1",
                        "publishedAt": "2026-06-03T10:00:00Z",
                        "source": {"name": "Example source"},
                    }
                ]
            },
        )

        with patch("main.requests.get", return_value=response) as mocked_get:
            items = main.fetch_latest_news("news-key", query="sports", page_size=2, language="en")

        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.title, "First title")
        self.assertEqual(item.description, "First description")
        self.assertEqual(item.content, "First content")
        self.assertEqual(item.url, "https://example.com/article-1")
        self.assertEqual(item.published_at, "2026-06-03T10:00:00Z")
        self.assertEqual(item.source, "Example source")
        mocked_get.assert_called_once()

    def test_fetch_latest_news_logs_distinct_http_errors(self) -> None:
        cases = [
            (401, "auth"),
            (403, "forbidden"),
            (429, "rate_limit"),
            (500, "server_error"),
            (418, "client_error"),
        ]

        for status_code, label in cases:
            with self.subTest(status_code=status_code):
                response = FakeResponse(status_code, {"message": "boom"})
                with patch("main.requests.get", return_value=response), patch("main._log") as mocked_log:
                    items = main.fetch_latest_news("news-key", query="sports", page_size=2, language="en")

                self.assertEqual(items, [])
                logged = " ".join(str(call.args[0]) for call in mocked_log.call_args_list)
                self.assertIn(f"[{label}]", logged)

    def test_summarize_news_with_gemini_uses_structured_output_and_retries(self) -> None:
        calls: list[dict[str, object]] = []
        sleep_mock = Mock()
        client_holder: dict[str, object] = {}

        class FakeModels:
            def __init__(self) -> None:
                self.attempts = 0

            def generate_content(self, *, model: str, contents: str, config: object) -> object:
                self.attempts += 1
                calls.append(
                    {
                        "model": model,
                        "contents": contents,
                        "config": config,
                    }
                )
                if self.attempts == 1:
                    raise RuntimeError("429 quota exceeded; please retry in 0.01s")
                return SimpleNamespace(
                    parsed={
                        "headline": "Short headline",
                        "summary": "Two sentence summary.",
                        "category": "Tech",
                    }
                )

        class FakeClient:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.models = FakeModels()
                client_holder["client"] = self

        fake_genai = SimpleNamespace(Client=FakeClient)
        fake_types = SimpleNamespace(
            GenerateContentConfig=FakeGenerateContentConfig,
            Schema=main.genai_types.Schema,
            Type=main.genai_types.Type,
        )

        with patch.object(main, "genai", fake_genai), patch.object(main, "genai_types", fake_types), patch.object(main.time, "sleep", sleep_mock):
            result = main.summarize_news_with_gemini(
                ["gemini-test-model"],
                "fake-api-key",
                "Title: Some title\nDescription: Some description",
            )

        self.assertEqual(result, {
            "headline": "Short headline",
            "summary": "Two sentence summary.",
            "category": "Tech",
            "source_url": "",
        })
        self.assertIn("client", client_holder)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["model"], "gemini-test-model")
        self.assertNotIn("СТРОГО JSON", str(calls[0]["contents"]))
        self.assertIn("Роль: ты новостной редактор.", str(calls[0]["contents"]))

        config = calls[0]["config"]
        self.assertEqual(getattr(config, "response_mime_type", None), "application/json")
        schema = getattr(config, "response_schema", None)
        self.assertIsNotNone(schema)
        self.assertEqual(getattr(schema, "type", None), main.genai_types.Type.OBJECT)
        self.assertIsNotNone(getattr(schema, "properties", None))
        self.assertEqual(sorted(getattr(schema, "required", [])), ["category", "headline", "summary"])
        sleep_mock.assert_called_once()
        self.assertAlmostEqual(float(sleep_mock.call_args.args[0]), 0.01, places=2)

    def test_write_results_json_writes_valid_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "results.json"
            main.write_results_json(
                str(path),
                [
                    {
                        "headline": "Headline",
                        "summary": "Summary",
                        "category": "Category",
                        "source_url": "https://example.com/item",
                    }
                ],
            )

            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)

        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["headline"], "Headline")
        self.assertEqual(data[0]["summary"], "Summary")
        self.assertEqual(data[0]["category"], "Category")
        self.assertEqual(data[0]["source_url"], "https://example.com/item")
        self.assertTrue(str(path).endswith(".json"))


if __name__ == "__main__":
    unittest.main()
