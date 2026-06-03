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
    from google import genai
    from google.genai import types as genai_types
except Exception as e:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    _GENAI_IMPORT_ERROR = e
else:
    _GENAI_IMPORT_ERROR = None


NEWS_API_URL = "https://newsapi.org/v2/everything"
TOP_HEADLINES_API_URL = "https://newsapi.org/v2/top-headlines"
DEFAULT_QUERY = ""
DEFAULT_PAGE_SIZE = 5
SEPARATOR = "-" * 40
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_RETRY_ATTEMPTS = 2


@dataclass(frozen=True)
class NewsItem:
    title: str
    description: str
    content: str
    url: str
    published_at: str
    source: str


def _build_gemini_response_schema() -> Any:
    if genai_types is None:
        return None

    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "headline": genai_types.Schema(type=genai_types.Type.STRING),
            "summary": genai_types.Schema(type=genai_types.Type.STRING),
            "category": genai_types.Schema(type=genai_types.Type.STRING),
        },
        required=["headline", "summary", "category"],
        property_ordering=["headline", "summary", "category"],
    )


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


def _load_gemini_keys_from_env() -> list[str]:
    """
    Загружает Gemini ключи из GEMINI_API_KEYS или GEMINI_API_KEY.

    Поддерживает как один ключ, так и список ключей через запятую в любом из
    этих параметров, чтобы не зависеть от имени переменной в .env.
    """
    gemini_api_keys = (os.getenv("GEMINI_API_KEYS") or "").strip()
    gemini_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()

    keys: list[str] = []
    seen: set[str] = set()

    # Сначала пробуем GEMINI_API_KEYS
    if gemini_api_keys:
        for key in _parse_comma_list(gemini_api_keys):
            if key not in seen:
                seen.add(key)
                keys.append(key)

    # Затем GEMINI_API_KEY
    if gemini_api_key:
        for key in _parse_comma_list(gemini_api_key):
            if key not in seen:
                seen.add(key)
                keys.append(key)

    return keys


def _contains_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text or ""))


def _slugify_topic_for_filename(topic: str) -> str:
    """
    Нормализует тему для имени файла.
    """
    normalized = re.sub(r"\s+", "_", (topic or "").strip().lower())
    normalized = re.sub(r"[^a-z0-9а-яё_-]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return "all_topics"
    return normalized[:50]


def build_results_file_path(topic: str) -> str:
    """
    Формирует путь к файлу в папке NEWS в формате NEWS/results_<topic>_<YYYYmmdd_HHMMSS>.json.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = _slugify_topic_for_filename(topic)
    return os.path.join("NEWS", f"results_{topic_slug}_{stamp}.json")


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
    news_api_key = (os.getenv("NEWS_API_KEY") or "").strip() or None
    gemini_keys = _load_gemini_keys_from_env()
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


def fetch_latest_news(
    news_api_key: str,
    query: str = DEFAULT_QUERY,
    page_size: int = DEFAULT_PAGE_SIZE,
    language: Optional[str] = "en",
    timeout_s: int = 20,
) -> list[NewsItem]:
    """
    Получает последние новости из NewsAPI и возвращает список NewsItem.
    Если query пустой, возвращает общие топ-новости без фильтра по теме.
    """
    normalized_query = (query or "").strip()
    page_size = max(1, min(page_size, 100))
    if normalized_query:
        url = NEWS_API_URL
        params = {
            "q": normalized_query,
            "pageSize": page_size,
            "sortBy": "publishedAt",
        }
        if language:
            params["language"] = language
    else:
        # Пустая тема: берем общие top headlines.
        url = TOP_HEADLINES_API_URL
        params = {
            "country": "us",
            "pageSize": page_size,
        }

    headers = {"X-Api-Key": news_api_key}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout_s)
    except requests.RequestException as e:
        _log(f"Ошибка сети при запросе NewsAPI: {e}")
        return []

    if resp.status_code != 200:
        try:
            details = resp.json()
        except Exception:
            details = {"text": resp.text[:500]}
        _log_newsapi_http_error(resp.status_code, details)
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


def _get_env_int(name: str, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    """
    Безопасно читает int из переменной окружения и ограничивает диапазон.
    При невалидном значении пишет лог и использует default.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            _log(f"Невалидное значение {name}={raw!r}, использую {default}.")
            value = default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _extract_last_json_object(text: str) -> Optional[dict[str, Any]]:
    """
    Ищет последний валидный JSON-объект в произвольном тексте.
    """
    decoder = json.JSONDecoder()
    last_obj: Optional[dict[str, Any]] = None
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            last_obj = obj
    return last_obj


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

    # 2) Fallback: попытаться найти последний валидный JSON-объект в тексте
    return _extract_last_json_object(text)


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


def _newsapi_error_label(status_code: int) -> str:
    if status_code == 401:
        return "auth"
    if status_code == 403:
        return "forbidden"
    if status_code == 429:
        return "rate_limit"
    if 400 <= status_code < 500:
        return "client_error"
    if 500 <= status_code < 600:
        return "server_error"
    return "unexpected"


def _log_newsapi_http_error(status_code: int, details: Any) -> None:
    label = _newsapi_error_label(status_code)
    _log(f"NewsAPI [{label}] вернул статус {status_code}. Детали: {details}")


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

    Возвращаемая структура:
    {
      "headline": "Краткий заголовок",
      "summary": "Саммари в 2-3 предложения",
      "category": "Категория"
    }
    """
    if not news_text.strip():
        return None

    # Длинные статьи съедают лимит токенов и могут обрезать JSON в ответе
    max_chars = _get_env_int("GEMINI_MAX_NEWS_CHARS", 8000, min_value=500, max_value=50000)
    if len(news_text) > max_chars:
        news_text = news_text[:max_chars] + "\n...[текст обрезан для API]"

    prompt = (
        "Роль: ты новостной редактор.\n"
        "Задача: прочитай новость и подготовь краткое структурированное описание.\n"
        "Требуемая структура:\n"
        "- headline: краткий заголовок\n"
        "- summary: саммари в 2-3 предложения\n"
        "- category: категория новости\n\n"
        "Новость:\n"
        f"{news_text}\n"
    )

    if genai is None or genai_types is None:
        _log("Gemini библиотека недоступна (google-generativeai не импортировалась).")
        return None

    client = genai.Client(api_key=gemini_api_key)
    # Structured output: API сам валидирует формат по response_schema.
    max_out = _get_env_int("GEMINI_MAX_OUTPUT_TOKENS", 2048, min_value=256, max_value=8192)
    cfg = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=_build_gemini_response_schema(),
        temperature=0.2,
        max_output_tokens=max(256, min(max_out, 8192)),
    )

    last_err: Exception | None = None
    retry_attempts = _get_env_int(
        "GEMINI_RETRY_ATTEMPTS",
        DEFAULT_GEMINI_RETRY_ATTEMPTS,
        min_value=1,
        max_value=5,
    )
    for candidate in model_names:
        attempts_left = retry_attempts
        while attempts_left > 0:
            try:
                resp = client.models.generate_content(model=candidate, contents=prompt, config=cfg)
                _log(f"Gemini: использую модель {candidate!r} для этой новости.")
                last_err = None
                break
            except Exception as e:
                last_err = e
                attempts_left -= 1
                if _looks_like_model_not_found(e):
                    _log(f"Модель {candidate!r} недоступна, пробую следующую...")
                    break

                if _looks_like_rate_limited(e):
                    # При лимитировании даём шанс повторить тот же ключ, а затем
                    # позволяем верхнему уровню переключиться на следующий ключ.
                    retry_seconds = _extract_retry_seconds(e) or min(5.0, 2.0 ** (retry_attempts - attempts_left))
                    _log(
                        "Gemini ограничил запросы (429/quota). "
                        f"Повтор через {retry_seconds:.2f}s, если попытки еще остались."
                    )
                    if attempts_left > 0:
                        time.sleep(retry_seconds)
                        continue
                    raise

                if _looks_like_invalid_api_key(e) or _looks_like_permission_denied(e):
                    # Это ошибка ключа/доступов — пусть верхний уровень попробует другой ключ
                    raise

                if attempts_left > 0:
                    retry_seconds = _extract_retry_seconds(e) or 1.0
                    _log(
                        f"Ошибка при вызове Gemini (модель {candidate!r}): {e}. "
                        f"Повтор через {retry_seconds:.2f}s."
                    )
                    time.sleep(retry_seconds)
                    continue

                _log(f"Ошибка при вызове Gemini (модель {candidate!r}): {e}")
                return None

        if last_err is None:
            break
    else:
        _log(f"Не удалось вызвать Gemini ни на одной модели. Последняя ошибка: {last_err}")
        return None

    data: Optional[dict[str, Any]] = None
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, dict):
        data = parsed

    if data is None:
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

    return {"headline": headline, "summary": summary, "category": category, "source_url": ""}


def write_results_json(path: str, items: list[dict[str, str]]) -> None:
    """
    Записывает весь результат в JSON-файл.
    """
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except OSError as e:
        _log(f"Ошибка записи в файл {path!r}: {e}")


def prompt_user_query(default_query: str = DEFAULT_QUERY) -> str:
    """
    Спрашивает тему новостей у пользователя.
    Пустой ввод означает режим без фильтра по теме.
    """
    prompt = "Введите тему новостей (Enter = без фильтра): "
    try:
        raw = input(prompt)
    except EOFError:
        return default_query

    value = (raw or "").strip()
    if value and _contains_cyrillic(value):
        _log(
            "Подсказка: тема введена на русском. NewsAPI чаще лучше работает с английскими ключевыми словами; "
            "для русской темы будет включен language='ru'."
        )
    return value if value else default_query


def choose_query_language(query: str) -> Optional[str]:
    """
    Подбирает language-параметр для NewsAPI.
    """
    normalized_query = (query or "").strip()
    if not normalized_query:
        return None
    if _contains_cyrillic(normalized_query):
        return "ru"
    return "en"


def prompt_user_page_size(default_page_size: int = DEFAULT_PAGE_SIZE) -> int:
    """
    Спрашивает количество новостей.
    Пустой ввод = default_page_size.
    """
    while True:
        prompt = f"Введите количество новостей (Enter = {default_page_size}): "
        try:
            raw = input(prompt)
        except EOFError:
            return default_page_size

        value = (raw or "").strip()
        if not value:
            return default_page_size
        try:
            parsed = int(value)
        except ValueError:
            _log("Нужно ввести целое число.")
            continue

        if parsed < 1:
            _log("Количество новостей должно быть >= 1.")
            continue
        if parsed > 100:
            _log("NewsAPI поддерживает до 100 новостей за запрос. Беру 100.")
            return 100
        return parsed


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
    if gemini_keys:
        for idx, key in enumerate(gemini_keys, start=1):
            key_preview = key[:10] + "..." if len(key) > 10 else key
            _log(f"  Ключ {idx}: {key_preview}")

    query = prompt_user_query()
    page_size = prompt_user_page_size()
    query_language = choose_query_language(query)
    results_file = build_results_file_path(query)
    if query:
        _log(f"Выбрана тема новостей: {query!r}")
    else:
        _log("Тема не выбрана: запускаю без фильтра по теме (top headlines).")
    _log(f"Запрошенное количество новостей: {page_size}")
    _log(f"Параметр language для NewsAPI: {query_language!r}")
    _log(f"Результат будет записан в новый файл: {results_file!r}")

    _log("Запрашиваю новости из NewsAPI...")
    news = fetch_latest_news(news_api_key, query=query, page_size=page_size, language=query_language)
    if not news:
        _log("Новости не получены (пусто или ошибка). Завершаю.")
        return 1

    _log(f"Получено новостей: {len(news)}. Начинаю саммаризацию...")

    results: list[dict[str, str]] = []
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
                # Ошибки ключа/доступа, rate limiting: пробуем следующий ключ
                if (_looks_like_invalid_api_key(e) or _looks_like_permission_denied(e) or _looks_like_rate_limited(e)):
                    _log(f"Проблема с Gemini ключом {key_idx}/{len(gemini_keys)}: {e}. Переключаюсь на следующий.")
                    continue
                _log(f"Неожиданная ошибка при работе с ключом {key_idx}/{len(gemini_keys)}: {e}")
                break

        if result is None:
            _log(f"[{idx}/{len(news)}] Пропуск: не удалось получить валидный JSON.")
            continue

        if item.url and not result.get("source_url"):
            result["source_url"] = item.url

        results.append(result)
        write_results_json(results_file, results)
        _log(f"[{idx}/{len(news)}] OK: сохранено в {results_file!r}.")

        # Маленькая пауза, чтобы не упираться в лимиты слишком агрессивно
        time.sleep(0.5)

    _log(f"Готово. Успешно обработано: {len(results)}/{len(news)}.")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())