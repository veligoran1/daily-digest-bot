import os
import re
import random
import asyncio
from datetime import datetime, timedelta

import pytz
import requests
from bs4 import BeautifulSoup
import telegram
from groq import Groq

# ─── КОНФИГУРАЦИЯ ───────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

os.environ['KAGGLE_API_TOKEN'] = os.environ.get('KAGGLE_API_TOKEN', '')

SOURCES = {
    'habr': [
        'https://habr.com/ru/flows/ai_and_ml/articles/',
        'https://habr.com/ru/hubs/maths/articles/',
        'https://habr.com/ru/hubs/physics/articles/',
        'https://habr.com/ru/hubs/python/articles/'
    ],
    'hackernoon': [
        'https://hackernoon.com/c/science',
        'https://hackernoon.com/c/machine-learning',
        'https://hackernoon.com/c/data-science',
        'https://hackernoon.com/c/programming'
    ],
    'arxiv': [
    'http://export.arxiv.org/api/query?search_query=cat:cs.LG+OR+cat:cs.AI+OR+cat:cs.CV+OR+cat:math.OC+OR+cat:stat.ML&sortBy=submittedDate&sortOrder=descending&max_results=20'
    ]
}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
]

# ─── GROQ API КЛИЕНТ ───────────────────────────────────────────

groq_client = Groq(api_key=GROQ_API_KEY)


def ask_llm(prompt: str, max_tokens: int = 1500) -> str:
    """Отправляет промпт в Groq API и возвращает ответ."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content


# ─── ПАРСИНГ ИСТОЧНИКОВ ────────────────────────────────────────

def parse_articles(url, days_back=0):
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    moscow_tz = pytz.timezone('Europe/Moscow')
    target_date = datetime.now(moscow_tz).date() - timedelta(days=days_back)
    posts = []

    try:
        if 'habr.com' in url:
            response = requests.get(url, headers=headers, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('article', class_='tm-articles-list__item')

            for article in articles:
                time_elem = article.find('time')
                if not time_elem or 'datetime' not in time_elem.attrs:
                    continue

                pub_datetime = datetime.fromisoformat(time_elem['datetime'].replace('Z', '+00:00'))
                if pub_datetime.date() != target_date:
                    continue

                title_elem = article.find('a', class_='tm-title__link')
                if not title_elem:
                    continue

                posts.append({
                    'title': title_elem.get_text(strip=True),
                    'link': 'https://habr.com' + title_elem['href'],
                    'datetime': pub_datetime.replace(tzinfo=None),
                    'date': pub_datetime.date(),
                    'source': url
                })

        elif 'hackernoon.com' in url:
            response = requests.get(url, headers=headers, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('article')

            for article in articles:
                date_p = article.find('p', class_='text-lightTextLight')
                if not date_p:
                    continue
                try:
                    date_text = date_p.get_text(strip=True)
                    pub_datetime = datetime.strptime(date_text, '%b %d, %Y')
                    if pub_datetime.date() != target_date:
                        continue

                    title_elem = article.find('h2') or article.find('h3')
                    link_elem = article.find('a')

                    if title_elem and link_elem:
                        link = link_elem['href'] if link_elem['href'].startswith('http') else 'https://hackernoon.com' + link_elem['href']
                        posts.append({
                            'title': title_elem.get_text(strip=True),
                            'link': link,
                            'datetime': pub_datetime,
                            'date': pub_datetime.date(),
                            'source': url
                        })
                except Exception:
                    pass

        elif 'arxiv.org' in url:
            import xml.etree.ElementTree as ET
            response = requests.get(url, headers=headers, timeout=30)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            root = ET.fromstring(response.text)

            for entry in root.findall('atom:entry', ns):
                published = entry.find('atom:published', ns).text[:10]
                pub_date = datetime.strptime(published, '%Y-%m-%d').date()
                if pub_date != target_date:
                    continue

                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                link = entry.find('atom:id', ns).text
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')

                posts.append({
                    'title': title,
                    'link': link,
                    'text': summary,
                    'datetime': datetime.combine(pub_date, datetime.min.time()),
                    'date': pub_date,
                    'source': url
                })
    except Exception as e:
        print(f"Ошибка парсинга {url}: {e}")

    return posts


def get_full_article(url):
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        soup = BeautifulSoup(response.content, 'html.parser')

        if 'habr.com' in url:
            article_body = soup.find('div', class_='article-formatted-body')
            if article_body:
                for tag in article_body(['script', 'style', 'code', 'pre']):
                    tag.decompose()
                paragraphs = article_body.find_all(['p', 'h2', 'h3', 'li'])
                text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
                return re.sub(r'\s+', ' ', text)

        elif 'hackernoon.com' in url:
            article_body = soup.find('div', class_='prose')
            if article_body:
                for tag in article_body(['script', 'style', 'code', 'pre']):
                    tag.decompose()
                paragraphs = article_body.find_all('p')
                text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])
                return re.sub(r'\s+', ' ', text)
    except Exception:
        pass
    return ""


def get_kaggle_competitions():
    """Получает соревнования Kaggle напрямую через HTTP API."""
    token = os.environ.get('KAGGLE_API_TOKEN', '')
    if not token:
        print("⚠️ KAGGLE_API_TOKEN не задан, пропускаем Kaggle")
        return []

    try:
        headers = {'Authorization': f'Bearer {token}'}
        r = requests.get(
            'https://www.kaggle.com/api/v1/competitions/list?page=1',
            headers=headers, timeout=30
        )
        r.raise_for_status()

        competitions = r.json()
        recent_comps = []
        now = datetime.now()

        for comp in competitions:
            enabled_str = comp.get('enabledDate', '')
            if not enabled_str:
                continue

            enabled_date = datetime.fromisoformat(enabled_str.replace('Z', '+00:00')).replace(tzinfo=None)
            days_since_start = (now - enabled_date).days
            team_count = comp.get('teamCount', 0)

            if days_since_start <= 30 and team_count > 50:
                recent_comps.append({
                    'title': comp.get('title', ''),
                    'link': f"https://www.kaggle.com/competitions/{comp.get('ref', '')}",
                    'text': f"{comp.get('description', '')}. Соревнование началось {enabled_date.date()}, участвует {team_count} команд.",
                    'teams': team_count,
                    'date': enabled_date.date(),
                    'datetime': enabled_date,
                    'days_since_start': days_since_start,
                    'source': 'kaggle'
                })

        recent_comps.sort(key=lambda x: x['teams'], reverse=True)
        return recent_comps[:5]

    except Exception as e:
        print(f"Ошибка Kaggle API: {e}")
        return []


# ─── ГЕНЕРАЦИЯ САММАРИ ЧЕРЕЗ CLAUDE ────────────────────────────

def get_summary_parts(posts, target_date):
    kaggle_posts = [p for p in posts if p.get('source') == 'kaggle']
    arxiv_posts = [p for p in posts if 'arxiv.org' in p['link']]
    blog_posts = [p for p in posts if 'habr.com' in p['link'] or 'hackernoon.com' in p['link']]

    date_str = target_date.strftime('%d.%m.%Y')
    result = {'annotation': '', 'kaggle': [], 'arxiv': '', 'blogs': ''}

    # 1. АННОТАЦИЯ
    all_titles = [f"- {p['title']}" for p in posts]
    annotation_prompt = f"""Твоя задача: написать подробную аннотацию дня за {date_str} НА РУССКОМ ЯЗЫКЕ.

Темы дня:
{chr(10).join(all_titles)}

ТРЕБОВАНИЯ:
- Объем: 6-8 предложений
- Язык: ТОЛЬКО русский!
- Структура:
  1. Общий контекст дня (1-2 предложения)
  2. Ключевые направления и тренды (2-3 предложения)
  3. Интересные темы и технологии (2-3 предложения)
  4. Общий вывод (1 предложение)

Пиши связно, без списков и маркеров. Текст должен читаться как единое целое."""

    result['annotation'] = ask_llm(annotation_prompt, max_tokens=600)

    # 2. KAGGLE
    for kp in kaggle_posts:
        kg_prompt = f"""Опиши соревнование Kaggle ОДНИМ-ДВУМЯ предложениями НА РУССКОМ ЯЗЫКЕ.

Название: {kp['title']}
Дата начала: {kp.get('date')}
Количество команд: {kp.get('teams')}
Описание: {kp['text']}

ТРЕБОВАНИЯ:
- 1-2 предложения, ТОЛЬКО русский
- Включить: дату начала, количество команд, суть задачи
- В конце добавь ссылку: {kp['link']}"""

        result['kaggle'].append(ask_llm(kg_prompt, max_tokens=200))

    # 3. ARXIV
    if arxiv_posts:
        arxiv_context = "\n\n".join([
            f"{i+1}. Название: {p['title']}\nТекст: {p.get('text', '')[:800]}\nСсылка: {p['link']}"
            for i, p in enumerate(arxiv_posts)
        ])
        arxiv_prompt = f"""Для КАЖДОЙ научной статьи создай саммари НА РУССКОМ ЯЗЫКЕ.

{arxiv_context}

ТРЕБОВАНИЯ для КАЖДОЙ статьи:
- 3-4 предложения, ТОЛЬКО русский
- Структура: что изучается, методы, результаты, значение

Формат:
1. [Саммари] - [ссылка]
2. [Саммари] - [ссылка]"""

        result['arxiv'] = ask_llm(arxiv_prompt, max_tokens=1500)

    # 4. БЛОГИ
    if blog_posts:
        blog_context = "\n\n".join([
            f"{i+1}. Название: {p['title']}\nТекст: {p.get('text', '')[:1500]}\nСсылка: {p['link']}"
            for i, p in enumerate(blog_posts)
        ])
        blog_prompt = f"""Для КАЖДОЙ статьи создай подробное саммари НА РУССКОМ ЯЗЫКЕ.

{blog_context}

ТРЕБОВАНИЯ для КАЖДОЙ статьи:
- 5-7 предложений, ТОЛЬКО русский
- Структура: тема, идеи, детали, польза

Формат:
1. [Саммари] - [ссылка]
2. [Саммари] - [ссылка]"""

        result['blogs'] = ask_llm(blog_prompt, max_tokens=3500)

    return result


# ─── ОТПРАВКА В TELEGRAM ───────────────────────────────────────

async def send_to_telegram(parts, target_date):
    bot = telegram.Bot(token=TELEGRAM_TOKEN)

    async def safe_send(text):
        """Разбивает длинные сообщения на части по 4000 символов."""
        if len(text) <= 4000:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
            await asyncio.sleep(1)
        else:
            chunks = []
            current = ""
            for paragraph in text.split('\n\n'):
                if len(current) + len(paragraph) + 2 > 3900:
                    chunks.append(current)
                    current = paragraph
                else:
                    current += ('\n\n' + paragraph if current else paragraph)
            if current:
                chunks.append(current)
            for chunk in chunks:
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=chunk)
                await asyncio.sleep(1)

    # 1. Аннотация
    await safe_send(f"📊 Digest за {target_date.strftime('%d.%m.%Y')}\n\n{parts['annotation']}")

    # 2. Kaggle
    if parts['kaggle']:
        msg = "🏆 СОРЕВНОВАНИЯ KAGGLE:\n\n"
        for i, kg in enumerate(parts['kaggle'], 1):
            msg += f"{i}. {kg}\n\n"
        await safe_send(msg)

    # 3. Arxiv
    if parts['arxiv']:
        await safe_send(f"📚 НАУЧНЫЕ СТАТЬИ:\n\n{parts['arxiv']}")

    # 4. Блоги
    if parts['blogs']:
        await safe_send(f"✍️ БЛОГИ:\n\n{parts['blogs']}")

    print("✅ Digest отправлен!")


# ─── ГЛАВНАЯ ФУНКЦИЯ ───────────────────────────────────────────

def main(days_back=1):
    print("🚀 Начинаем сбор дайджеста...")

    # Сбор источников
    all_sources = []
    for category in SOURCES.values():
        all_sources.extend(category)

    all_posts = []
    for url in all_sources:
        print(f"  Парсинг: {url[:50]}...")
        posts = parse_articles(url, days_back=days_back)
        all_posts.extend(posts)

    # Kaggle
    print("  Загрузка Kaggle соревнований...")
    all_posts.extend(get_kaggle_competitions())

    # Дедупликация
    unique_posts = {post['link']: post for post in all_posts}
    all_posts = list(unique_posts.values())

    # Ограничение Arxiv
    arxiv = [p for p in all_posts if 'arxiv.com' in p['link']][:5]
    others = [p for p in all_posts if 'arxiv.com' not in p['link']]
    all_posts = others + arxiv

    # Загрузка текстов
    for post in all_posts:
        if 'text' not in post or not post['text']:
            post['text'] = get_full_article(post['link'])

    # Фильтрация пустых
    all_posts = [p for p in all_posts if p.get('source') == 'kaggle' or len(p.get('text', '')) > 100]

    # Очистка timezone и сортировка
    for post in all_posts:
        if hasattr(post['datetime'], 'tzinfo') and post['datetime'].tzinfo:
            post['datetime'] = post['datetime'].replace(tzinfo=None)
    all_posts.sort(key=lambda x: x['datetime'])

    print(f"\n📊 Найдено: {len(all_posts)} статей")

    if not all_posts:
        print("⚠️ Статей не найдено, дайджест не отправляется.")
        return

    # Генерация саммари
    moscow_tz = pytz.timezone('Europe/Moscow')
    target_date = datetime.now(moscow_tz).date() - timedelta(days=days_back)
    print("🤖 Генерация саммари через Groq API...")
    parts = get_summary_parts(all_posts, target_date)

    # Отправка в Telegram
    print("📤 Отправка в Telegram...")
    asyncio.run(send_to_telegram(parts, target_date))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-back", type=int, default=1)
    args = parser.parse_args()
    main(days_back=args.days_back)