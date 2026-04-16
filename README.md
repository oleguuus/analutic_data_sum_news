# NewsAPI → Gemini Summarizer

CLI-скрипт для сбора новостей, их краткой структурированной обработки через Gemini и сохранения результата в отдельный файл.

## Что делает скрипт

1. Спрашивает у пользователя тему новостей.
2. Если тема не указана, берет общие top headlines без фильтра.
3. Спрашивает количество новостей.
4. Загружает новости из NewsAPI.
5. Отправляет каждую новость в Google Gemini для саммаризации.
6. Добавляет в ответ ссылку на первоисточник.
7. Сохраняет результат в отдельный файл внутри папки `NEWS`.

## Формат результата

Каждый запуск создает новый файл вида:

`NEWS/results_<topic>_<YYYYmmdd_HHMMSS>.txt`

Если тема не указана, используется имя с префиксом `all_topics`.

Пример записи внутри файла:

```text
[Бизнес] Краткий заголовок
Саммари: Краткое описание новости.
Источник: https://example.com/article
----------------------------------------
```

## Требования

- Python 3.10+
- Ключи API:
  - `NEWS_API_KEY` для NewsAPI
  - `GEMINI_API_KEY` или `GEMINI_API_KEYS` для Gemini

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Если PowerShell блокирует запуск скриптов, один раз выполни:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Настройка ключей

```powershell
Copy-Item .env.example .env
notepad .env
```

Заполни файл `.env` своими ключами:

```env
NEWS_API_KEY=your_newsapi_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Можно также указать несколько ключей или модель:

```env
GEMINI_API_KEYS=key1,key2,key3
GEMINI_API_KEY=key1,key2,key3
GEMINI_MODEL=gemini-3.1-flash
```

Скрипт принимает список ключей как в `GEMINI_API_KEYS`, так и в `GEMINI_API_KEY`.

## Запуск

```powershell
python .\main.py
```

Во время запуска скрипт спросит:

1. тему новостей;
2. количество новостей.

Если просто нажать Enter:

- тема останется пустой, и скрипт возьмет общие top headlines;
- количество новостей будет равно 5.

## Русская тема

Если тема введена на русском, скрипт автоматически переключает `language=ru` для NewsAPI и пишет подсказку в лог.

## Модели Gemini

Если в логах появляется сообщение о том, что модель недоступна, это означает, что для текущего ключа или региона модель не активна. Скрипт автоматически пробует fallback-модели и пишет, какая модель реально использовалась.

## Устранение проблем

- Если видишь `429` или `Quota exceeded`, у ключа Gemini не включена квота или биллинг.
- Если новости не сохраняются, проверь наличие папки `NEWS` и права на запись.

## Пример запроса на тему "CAR"

```text
[Automotive] 2008 Chevrolet Corvette Z06 Track Car at No Reserve
Саммари: This 2008 Chevrolet Corvette Z06 was extensively modified for track use under previous ownership. In 2025, it received approximately $40,000 in upgrades, including a rebuilt 7.0-liter LS7 V8 engine, an RPS twin-disc clutch, overhauled brakes, and new sway bars.
Источник: https://bringatrailer.com/listing/2008-chevrolet-corvette-z06-76/
----------------------------------------
[Economy] Gig workers trapped by soaring gas prices
Саммари: Gig workers, including rideshare drivers and food couriers, are facing significant financial strain due to soaring gas prices, which places the entire burden of increased costs on them. Many feel trapped by this situation, exacerbated by ongoing global conflicts.
Источник: https://finance.yahoo.com/economy/articles/no-choice-gig-workers-trapped-122500268.html
----------------------------------------
```



