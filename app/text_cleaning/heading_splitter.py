"""
Модуль для разделения Markdown-текста на фрагменты по заголовкам.

Определяет иерархию заголовков (H1-H6) и делит текст на секции,
каждая из которых наследует иерархию родительских заголовков.
"""
import re
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

# Паттерн для распознавания Markdown-заголовков (# ... ######)
HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


@dataclass
class HeadingSection:
    """Фрагмент текста с иерархией заголовков."""
    heading_hierarchy: List[str]
    """Иерархия заголовков от верхнего уровня к текущему, например: [\"Глава 1\", \"Раздел 1.1\"]"""
    text: str
    """Текст секции (включая сам заголовок)"""
    level: int
    """Уровень текущего заголовка (1-6), 0 для текста без заголовка"""


def parse_heading_sections(markdown_text: str) -> List[HeadingSection]:
    """
    Разбивает Markdown-текст на секции по заголовкам, сохраняя иерархию.

    Логика:
    - Текст до первого заголовка попадает в секцию с пустой иерархией (level=0).
    - Каждый заголовок начинает новую секцию.
    - Иерархия строится по вложенности: H2 после H1 наследует H1,
      H3 после H2 наследует H1 → H2, и т.д.
    - Текст между текущим заголовком и следующим принадлежит текущей секции.

    Args:
        markdown_text: Исходный Markdown-текст.

    Returns:
        Список секций HeadingSection с иерархией и текстом.
    """
    if not markdown_text or not markdown_text.strip():
        return []

    lines = markdown_text.split('\n')

    # Определяем позиции заголовков
    heading_positions = []  # (line_index, level, heading_text)
    for i, line in enumerate(lines):
        match = HEADING_PATTERN.match(line.strip())
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            heading_positions.append((i, level, heading_text))

    # Если заголовков нет — возвращаем весь текст как одну секцию
    if not heading_positions:
        return [HeadingSection(
            heading_hierarchy=[],
            text=markdown_text.strip(),
            level=0,
        )]

    sections: List[HeadingSection] = []

    # Текст до первого заголовка
    first_heading_line = heading_positions[0][0]
    preamble = '\n'.join(lines[:first_heading_line]).strip()
    if preamble:
        sections.append(HeadingSection(
            heading_hierarchy=[],
            text=preamble,
            level=0,
        ))

    # Каждый заголовок — отдельная секция. Текст от заголовка до следующего принадлежит ему.
    hierarchy_stack: List[tuple] = []

    for pos_idx, (line_idx, level, heading_text) in enumerate(heading_positions):
        # Обновляем стек иерархии.
        # Убираем из стека все заголовки с уровнем >= текущего
        while hierarchy_stack and hierarchy_stack[-1][0] >= level:
            hierarchy_stack.pop()

        # Добавляем текущий заголовок в стек
        hierarchy_stack.append((level, heading_text))

        # Формируем иерархию из стека
        current_hierarchy = [h[1] for h in hierarchy_stack]

        # Конец секции — начало следующего заголовка или конец текста
        if pos_idx + 1 < len(heading_positions):
            next_line_idx = heading_positions[pos_idx + 1][0]
        else:
            next_line_idx = len(lines)

        section_text = '\n'.join(lines[line_idx:next_line_idx]).strip()

        sections.append(HeadingSection(
            heading_hierarchy=current_hierarchy,
            text=section_text,
            level=level,
        ))

    logger.debug(
        f"Parsed {len(sections)} heading sections from markdown "
        f"({len(heading_positions)} headings found)"
    )

    return sections


def build_full_category_path(
    doc_categories: List[str],
    heading_hierarchy: List[str],
) -> List[str]:
    """
    Строит полный путь категорий: категории документа + иерархия заголовков.

    Args:
        doc_categories: Категории документа, например [\"Документация\", \"API\"].
        heading_hierarchy: Иерархия заголовков секции, например [\"Глава 1\", \"Раздел 1.1\"].

    Returns:
        Полный путь, например [\"Документация\", \"API\", \"Глава 1\", \"Раздел 1.1\"].
    """
    return list(doc_categories) + list(heading_hierarchy)
