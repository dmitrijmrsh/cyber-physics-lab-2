"""
Скрипт для отправки запросов к модели Qwen2.5:0.5B через сервер Ollama.
Генерирует отчёт инференса в формате Markdown-таблицы.
"""

import requests
import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"

QUERIES = [
    "Что такое искусственный интеллект? Объясни простыми словами.",
    "Назови три самых популярных языка программирования и области их применения.",
    "Чем машинное обучение отличается от классического программирования?",
    "Что такое нейронная сеть? Объясни в одном предложении.",
    "Какова столица России и сколько в ней жителей?",
    "Объясни, что такое большие языковые модели (LLM).",
    "Что такое градиентный спуск в контексте обучения нейронных сетей?",
    "Приведи пример задачи, которую решает обучение с учителем.",
    "В чём разница между глубоким обучением и машинным обучением?",
    "Что такое переобучение модели и как с ним бороться?",
]


def send_query(query: str, model: str = MODEL_NAME, url: str = OLLAMA_URL) -> str:
    """
    Отправляет один запрос к серверу Ollama и возвращает текстовый ответ модели.

    Args:
        query: Текст запроса к LLM.
        model: Название модели (по умолчанию qwen2.5:0.5b).
        url:   URL эндпоинта Ollama (по умолчанию http://localhost:11434/api/generate).

    Returns:
        Текст ответа модели в виде строки.

    Raises:
        requests.exceptions.ConnectionError: Если сервер Ollama недоступен.
        requests.exceptions.HTTPError:       Если сервер вернул HTTP-ошибку.
    """
    payload = {
        "model": model,
        "prompt": query,
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def run_inference(queries: list[str]) -> list[dict]:
    """
    Прогоняет список запросов через LLM и собирает пары (запрос, ответ).

    Args:
        queries: Список строк-запросов к модели.

    Returns:
        Список словарей вида {"query": str, "response": str}.
    """
    results = []
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Sending: {query[:60]}...")
        answer = send_query(query)
        results.append({"query": query, "response": answer})
        print(f"         Got {len(answer)} chars back.\n")
    return results


def build_markdown_report(results: list[dict], model: str = MODEL_NAME) -> str:
    """
    Формирует Markdown-отчёт инференса с двумя столбцами: запрос и ответ LLM.

    Args:
        results: Список пар {"query": str, "response": str}.
        model:   Название использованной модели (отображается в заголовке).

    Returns:
        Строка с полным Markdown-документом.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Отчёт инференса — {model}",
        f"\n**Дата:** {timestamp}  ",
        f"**Модель:** `{model}`  ",
        f"**Сервер:** `http://localhost:11434`  ",
        f"**Количество запросов:** {len(results)}\n",
        "---\n",
        "| # | Запрос к LLM | Ответ LLM |",
        "|---|---|---|",
    ]
    for i, item in enumerate(results, 1):
        query = item["query"].replace("|", "\\|")
        response = item["response"].replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {i} | {query} | {response} |")

    return "\n".join(lines) + "\n"


def save_report(report: str, path: str = "report.md") -> None:
    """
    Сохраняет текстовый отчёт в файл.

    Args:
        report: Содержимое отчёта в виде строки.
        path:   Путь к выходному файлу (по умолчанию report.md).
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {path}")


def main() -> None:
    """
    Точка входа: запускает инференс по всем запросам,
    выводит результаты в консоль и сохраняет Markdown-отчёт.
    """
    print(f"Starting inference with model '{MODEL_NAME}' on {OLLAMA_URL}\n")
    results = run_inference(QUERIES)

    print("\n=== INFERENCE REPORT ===\n")
    for i, item in enumerate(results, 1):
        print(f"Query {i}: {item['query']}")
        print(f"Response: {item['response']}\n")

    report = build_markdown_report(results)
    save_report(report)


if __name__ == "__main__":
    main()
