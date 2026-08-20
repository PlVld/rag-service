"""
Нативные утилиты, лежащие внутри проекта и переносимые вместе с ним.

Пути считаются от корня проекта, поэтому не зависят от того, куда скопирована папка.
Если утилиты нет, возвращаем None — вызывающий код откатывается на системный PATH
(так работает Linux/Docker, где ставится системный пакет).
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESSERACT_DIR = PROJECT_ROOT / "tesseract"
# Раскладка Windows-сборок poppler: внутри архива лежит каталог Library/bin
POPPLER_BIN_DIR = PROJECT_ROOT / "poppler" / "Library" / "bin"


def bundled_tesseract_cmd() -> Optional[str]:
    """Путь к исполняемому файлу Tesseract из папки проекта."""
    exe_name = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    exe_path = TESSERACT_DIR / exe_name
    return str(exe_path) if exe_path.is_file() else None


def bundled_tessdata_dir() -> Optional[str]:
    """Каталог с языковыми файлами (*.traineddata) из папки проекта."""
    tessdata = TESSERACT_DIR / "tessdata"
    return str(tessdata) if tessdata.is_dir() else None
