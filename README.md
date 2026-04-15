# NewsAPI → Gemini Summarizer

Скрипт автоматически:
1) скачивает 3–5 последних новостей по запросу `technology` из NewsAPI;
2) отправляет каждую новость в Google Gemini (`gemini-3.1-flash`) для саммаризации;
3) получает ответ **строго в JSON**;
4) сохраняет результат в `results.txt` в читаемом формате.

## Требования
- Python 3.10+
- Ключи API:
  - `NEWS_API_KEY` (NewsAPI)
  - `GEMINI_API_KEY` (Google Gemini)

## Установка (Windows / PowerShell)
Создайте виртуальное окружение и установите зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Настройка ключей
Скопируйте пример и заполните ключи:

```powershell
Copy-Item .env.example .env
notepad .env
```

## Запуск

```powershell
python .\main.py
```

Результат будет сохранён в `results.txt`.

## Важно про Gemini квоты
Если при запуске видите ошибки `429` / `Quota exceeded` или в логах написано, что лимит равен 0, это значит, что для вашего `GEMINI_API_KEY` не включена квота/биллинг на Gemini API.
В этом случае саммаризация не сможет выполняться, пока вы не активируете доступ в Google AI Studio / Gemini API.
