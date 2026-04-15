from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import requests
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
except Exception as e:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    GenerationConfig = None  # type: ignore[assignment]
    _GENAI_IMPORT_ERROR = e
else:
    _GENAI_IMPORT_ERROR = None


NEWS_API_URL = "https://newsapi.org/v2/everything"
DEFAULT_QUERY = "technology"
DEFAULT_PAGE_SIZE = 5
RESULTS_FILE = "results.txt"
SEPARATOR = "-" * 40
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash"


@dataclass(frozen=True)
class NewsItem:
    title: str
    description: str
    content: str
    url: str
    published_at: str
    source: str


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(msg: str) -> None:
    print(f"[{_now_iso()}] {msg}")


def _ensure_utf8_console() -> None:
    """
    На Windows консоль часто использует кодировку, из-за чего кириллица может отображаться как "����".
    Здесь пытаемся принудительно переключить stdout/stderr на UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _parse_comma_list(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def load_keys() -> tuple[Optional[str], list[str], str]:
    """
    Загружает ключи из .env.

    Ожидаемые переменные окружения:
    - NEWS_API_KEY
    - GEMINI_API_KEY или GEMINI_API_KEYS (список через запятую)
    Опционально:
    - GEMINI_MODEL (по умолчанию gemini-3.1-flash)
    """
    load_dotenv()
    news_api_key = os.getenv("NEWS_API_KEY")
    gemini_keys: list[str] = []
    gemini_keys_raw = (os.getenv("GEMINI_API_KEYS") or "").strip()
    if gemini_keys_raw:
        gemini_keys = _parse_comma_list(gemini_keys_raw)
    else:
        single = (os.getenv("GEMINI_API_KEY") or "").strip()
        if single:
            gemini_keys = [single]
    gemini_model = (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    return news_api_key, gemini_keys, gemini_model


def _pick_fallback_model(preferred: str) -> list[str]:
    """
    На некоторых ключах/регионах конкретная модель может быть недоступна.
    Возвращаем список кандидатов: сначала preferred, затем безопасные flash-альтернативы.
    """
    candidates = [preferred]
    for name in (
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
    ):
        if name not in candidates:
            candidates.append(name)
    return candidates


def configure_gemini(gemini_api_key: str, model_name: str) -> Any:
    """
    Настраивает клиент Google Gemini и возвращает объект модели.
    """
    if genai is None:
        raise RuntimeError(
            "Библиотека google-generativeai не импортировалась. "
            "Проверьте установку зависимостей. "
            f"Ошибка импорта: {_GENAI_IMPORT_ERROR!r}"
        )

    genai.configure(api_key=gemini_api_key)
    last_error: Exception | None = None
    for candidate in _pick_fallback_model(model_name):
        try:
            _log(f"Использую Gemini модель: {candidate!r}")
            return genai.GenerativeModel(candidate)
        except Exception as e:
            last_error = e
            _log(f"Не удалось создать модель {candidate!r}: {e}")

    raise RuntimeError(f"Не удалось настроить Gemini модель. Последняя ошибка: {last_error}")


def fetch_latest_news(
    news_api_key: str,
    query: str = DEFAULT_QUERY,
    page_size: int = DEFAULT_PAGE_SIZE,
    language: str = "en",
    timeout_s: int = 20,
) -> list[NewsItem]:
    """
    Получает последние новости из NewsAPI и возвращает список NewsItem.
    Забираем 3–5 новостей (page_size) по запросу query.
    """
    params = {
        "q": query,
        "pageSize": max(1, min(page_size, 5)),
        "sortBy": "publishedAt",
        "language": language,
    }
    headers = {"X-Api-Key": news_api_key}

    try:
        resp = requests.get(NEWS_API_URL, params=params, headers=headers, timeout=timeout_s)
    except requests.RequestException as e:
        _log(f"Ошибка сети при запросе NewsAPI: {e}")
        return []

    if resp.status_code != 200:
        try:
            details = resp.json()
        except Exception:
            details = {"text": resp.text[:500]}
        _log(f"NewsAPI вернул статус {resp.status_code}. Детали: {details}")
        return []

    try:
        payload = resp.json()
    except Exception as e:
        _log(f"Не удалось распарсить ответ NewsAPI как JSON: {e}. Текст: {resp.text[:500]}")
        return []

    articles = payload.get("articles")
    if not isinstance(articles, list):
        _log(f"Неожиданный формат NewsAPI (articles): {type(articles)}")
        return []

    items: list[NewsItem] = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        title = str(a.get("title") or "").strip()
        description = str(a.get("description") or "").strip()
        content = str(a.get("content") or "").strip()
        url = str(a.get("url") or "").strip()
        published_at = str(a.get("publishedAt") or "").strip()
        source_obj = a.get("source") or {}
        source = ""
        if isinstance(source_obj, dict):
            source = str(source_obj.get("name") or "").strip()

        # Если совсем пусто — пропускаем
        if not (title or description or content):
            continue

        items.append(
            NewsItem(
                title=title,
                description=description,
                content=content,
                url=url,
                published_at=published_at,
                source=source,
            )
        )

    return items


def build_news_text(item: NewsItem) -> str:
    """
    Собирает единый текст новости для отправки в LLM.
    """
    parts: list[str] = []
    if item.title:
        parts.append(f"Title: {item.title}")
    if item.description:
        parts.append(f"Description: {item.description}")
    if item.content:
        parts.append(f"Content: {item.content}")
    if item.source:
        parts.append(f"Source: {item.source}")
    if item.published_at:
        parts.append(f"PublishedAt: {item.published_at}")
    if item.url:
        parts.append(f"URL: {item.url}")
    return "\n".join(parts).strip()


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}$", flags=re.MULTILINE)


def _safe_json_loads(text: str) -> Optional[dict[str, Any]]:
    """
    Пытается распарсить JSON максимально безопасно.
    Возвращает словарь или None.
    """
    text = (text or "").strip()
    if not text:
        return None

    # 1) Прямая попытка
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # 2) Fallback: попытаться вытащить JSON-объект из конца ответа
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _looks_like_model_not_found(err: Exception) -> bool:
    msg = str(err).lower()
    return ("404" in msg and "not found" in msg) or ("is not found" in msg) or ("model" in msg and "not found" in msg)


def _looks_like_rate_limited(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg or "quota exceeded" in msg or "rate limit" in msg


def _looks_like_invalid_api_key(err: Exception) -> bool:
    msg = str(err).lower()
    return "api key not valid" in msg or "api_key_invalid" in msg or "invalid api key" in msg


def _looks_like_permission_denied(err: Exception) -> bool:
    msg = str(err).lower()
    return "permission" in msg and ("denied" in msg or "forbidden" in msg)


def _extract_retry_seconds(err: Exception) -> Optional[float]:
    """
    Пытаемся вытащить подсказку вида 'Please retry in 15.33s' из текста ошибки.
    """
    msg = str(err)
    m = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", msg, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def summarize_news_with_gemini(
    model_names: list[str],
    gemini_api_key: str,
    news_text: str,
) -> Optional[dict[str, str]]:
    """
    Отправляет текст новости в Gemini и возвращает строго структурированный результат.

    Ожидаемый JSON:
    {
      "headline": "Краткий заголовок",
      "summary": "Саммари в 2-3 предложения",
      "category": "Категория"
    }
    """
    if not news_text.strip():
        return None

    prompt = (
        "Ты — новостной редактор. Прочитай новость и верни СТРОГО JSON (application/json), "
        "без markdown, без пояснений, без лишних ключей.\n\n"
        "Требуемая структура JSON:\n"
        '{\"headline\":\"Краткий заголовок\",\"summary\":\"Саммари в 2-3 предложения\",\"category\":\"Категория\"}\n\n'
        "Новость:\n"
        f"{news_text}\n"
    )

    if genai is None or GenerationConfig is None:
        _log("Gemini библиотека недоступна (google-generativeai не импортировалась).")
        return None

    genai.configure(api_key=gemini_api_key)
    cfg = GenerationConfig(
        response_mime_type="application/json",
        temperature=0.2,
        max_output_tokens=400,
    )

    last_err: Exception | None = None
    for candidate in model_names:
        attempts_left = 2
        while attempts_left > 0:
            try:
                model = genai.GenerativeModel(candidate)
                resp = model.generate_content(prompt, generation_config=cfg)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if _looks_like_model_not_found(e):
                    _log(f"Модель {candidate!r} недоступна, пробую следующую...")
                    break

                if _looks_like_rate_limited(e):
                    retry_s = _extract_retry_seconds(e)
                    if retry_s is None:
                        retry_s = 5.0
                    retry_s = max(1.0, min(retry_s, 20.0))
                    attempts_left -= 1
                    _log(
                        "Gemini ограничил запросы (429/quota). "
                        f"Подожду {retry_s:.1f}s и попробую ещё раз... "
                        "Если лимит = 0, нужно включить квоту/биллинг в Google AI Studio."
                    )
                    time.sleep(retry_s)
                    continue

                if _looks_like_invalid_api_key(e) or _looks_like_permission_denied(e):
                    # Это ошибка ключа/доступов — пусть верхний уровень попробует другой ключ
                    raise

                _log(f"Ошибка при вызове Gemini (модель {candidate!r}): {e}")
                return None

        if last_err is None:
            break
    else:
        _log(f"Не удалось вызвать Gemini ни на одной модели. Последняя ошибка: {last_err}")
        return None

    raw_text = ""
    try:
        raw_text = str(getattr(resp, "text", "") or "").strip()
    except Exception:
        raw_text = ""

    data = _safe_json_loads(raw_text)
    if data is None:
        _log(f"Gemini вернул невалидный JSON. Ответ (обрезан): {raw_text[:300]!r}")
        return None

    headline = str(data.get("headline") or "").strip()
    summary = str(data.get("summary") or "").strip()
    category = str(data.get("category") or "").strip()

    if not (headline and summary and category):
        _log(f"JSON от Gemini не содержит обязательных полей. Получено: {data}")
        return None

    return {"headline": headline, "summary": summary, "category": category}


def append_result(path: str, item: dict[str, str]) -> None:
    """
    Добавляет один результат в TXT-файл в читаемом формате.
    """
    category = item.get("category", "").strip() or "Без категории"
    headline = item.get("headline", "").strip() or "Без заголовка"
    summary = item.get("summary", "").strip() or ""

    block = f"[{category}] {headline}\nСаммари: {summary}\n{SEPARATOR}\n"
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)
    except OSError as e:
        _log(f"Ошибка записи в файл {path!r}: {e}")


def main() -> int:
    _ensure_utf8_console()
    news_api_key, gemini_keys, gemini_model = load_keys()

    if not news_api_key:
        _log("Не найден NEWS_API_KEY в .env (или переменных окружения).")
        return 2
    if not gemini_keys:
        _log("Не найден GEMINI_API_KEY или GEMINI_API_KEYS в .env (или переменных окружения).")
        return 2

    model_candidates = _pick_fallback_model(gemini_model)
    _log(f"Кандидаты моделей Gemini: {model_candidates}")
    _log(f"Gemini ключей загружено: {len(gemini_keys)}")

    _log(f"Запрашиваю новости из NewsAPI (query={DEFAULT_QUERY!r})...")
    news = fetch_latest_news(news_api_key, query=DEFAULT_QUERY, page_size=DEFAULT_PAGE_SIZE)
    if not news:
        _log("Новости не получены (пусто или ошибка). Завершаю.")
        return 1

    _log(f"Получено новостей: {len(news)}. Начинаю саммаризацию...")

    processed = 0
    for idx, item in enumerate(news, start=1):
        title_preview = (item.title or item.description or "")[:80]
        _log(f"[{idx}/{len(news)}] Обрабатываю: {title_preview!r}")

        news_text = build_news_text(item)
        result: Optional[dict[str, str]] = None
        for key_idx, key in enumerate(gemini_keys, start=1):
            try:
                if len(gemini_keys) > 1:
                    _log(f"Gemini ключ {key_idx}/{len(gemini_keys)}: пробую запрос...")
                result = summarize_news_with_gemini(model_candidates, key, news_text)
                if result is not None:
                    break
            except Exception as e:
                # Ошибки ключа/доступа: пробуем следующий ключ
                if _looks_like_invalid_api_key(e) or _looks_like_permission_denied(e):
                    _log(f"Проблема с Gemini ключом {key_idx}/{len(gemini_keys)}: {e}. Переключаюсь на следующий.")
                    continue
                _log(f"Неожиданная ошибка при работе с ключом {key_idx}/{len(gemini_keys)}: {e}")
                break

        if result is None:
            _log(f"[{idx}/{len(news)}] Пропуск: не удалось получить валидный JSON.")
            continue

        append_result(RESULTS_FILE, result)
        processed += 1
        _log(f"[{idx}/{len(news)}] OK: сохранено в {RESULTS_FILE!r}.")

        # Маленькая пауза, чтобы не упираться в лимиты слишком агрессивно
        time.sleep(0.5)

    _log(f"Готово. Успешно обработано: {processed}/{len(news)}.")
    return 0 if processed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

