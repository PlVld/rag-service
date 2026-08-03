import re

def normalize_for_embedding(text: str, source_format: str = 'text', category_path: str = '') -> str:
    """
    Подготавливает текст для векторизации (эмбеддингов).

    - Заменяет Markdown-ссылки на текст анкора
    - Удаляет Markdown-разметку (#, **, *, `)
    - Преобразует таблицы в линейный формат
    - Приводит к нижнему регистру
    - Нормализует пробелы и переносы строк
    - Оставляет базовую пунктуацию (.,!?;:()"'-)

    Args:
        text: Исходный текст (обычно Markdown или plain text)
        source_format: 'text', 'markdown', 'code', 'html', 'pdf', 'docx' и т.д.
        category_path: Путь категории, добавляется в начало текста для улучшения семантики

    Returns:
        Текст, готовый для передачи в encode_text()
    """
    if not text:
        return ""

    # Для кода – минимальная обработка
    if source_format == 'code':
        return text.strip()

    # Добавляем путь категории в начало текста
    if category_path:
        text = f"{category_path}. {text}"

    # 1. Замена ссылок [текст](url) -> текст
    text = re.sub(r'\[([^]]+)]\([^)]+\)', r'\1', text)

    # 2. Удаление Markdown-символов (заголовки, жирный, курсив, код)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  # # Заголовок
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **жирный**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # *курсив*
    text = re.sub(r'`([^`]+)`', r'\1', text)  # `код`

    # 3. Преобразование таблиц из Markdown в линейный вид
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Если строка похожа на Markdown-таблицу (начинается с | и содержит |)
        if line.strip().startswith('|') and '|' in line:
            # Убираем начальный и конечный |, заменяем | на пробел
            line = line.strip('|')
            line = re.sub(r'\s*\|\s*', ' ', line)
            # Удаляем строки-разделители (| --- | --- |)
            if re.match(r'^[\s\-:|]+$', line):
                continue
        cleaned_lines.append(line)
    text = ' '.join(cleaned_lines)

    # 4. Приведение к нижнему регистру
    text = text.lower()

    # 5. Нормализация пробелов и переносов строк
    text = re.sub(r'[ \t]+', ' ', text)  # множественные пробелы/табы -> один пробел
    text = re.sub(r'\n+', ' ', text)  # переносы строк -> пробел

    # 6. Удаление нежелательных символов (оставляем буквы, цифры, базовую пунктуацию)
    text = re.sub(r'[^\w\s.,!?;:()"\'\-]', ' ', text)

    # 7. Финальная очистка пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    return text