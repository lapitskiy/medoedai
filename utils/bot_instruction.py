import os

INSTRUCTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'content', 'telegram_instructions')

def load_telegram_instruction(name: str) -> str:
    """Загружает HTML-инструкцию из файла. Если файла нет, возвращает сообщение об ошибке."""
    filepath = os.path.join(INSTRUCTIONS_DIR, f"{name}.html")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        return f"❌ Ошибка: Файл инструкции {name}.html не найден."
