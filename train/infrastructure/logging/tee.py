from __future__ import annotations

import os
import sys
from typing import IO


class TailTruncatingTee:
    """Дублирует вывод в stdout и файл с ограничением размера.

    Если файл превышает max_bytes, оставляет только последние max_bytes данных.
    """

    def __init__(self, file_path: str, max_bytes: int = 100 * 1024 * 1024, also_stderr: bool = False):
        self.file_path = file_path
        self.max_bytes = int(max_bytes)
        self.also_stderr = bool(also_stderr)
        # Оригинальные потоки
        self._orig_stdout: IO[str] = sys.stdout
        self._orig_stderr: IO[str] = sys.stderr
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Открываем файл в режиме добавления в текстовом формате с буферизацией
        self._fh = open(self.file_path, 'a', encoding='utf-8', buffering=1)

    def write(self, data: str) -> int:
        # Пишем в оригинальные потоки
        try:
            self._orig_stdout.write(data)
        except Exception:
            pass
        if self.also_stderr:
            try:
                self._orig_stderr.write(data)
            except Exception:
                pass
        # Пишем в файл
        try:
            self._fh.write(data)
            self._fh.flush()
            self._truncate_if_needed()
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        try:
            self._orig_stdout.flush()
        except Exception:
            pass
        if self.also_stderr:
            try:
                self._orig_stderr.flush()
            except Exception:
                pass
        try:
            self._fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def _truncate_if_needed(self) -> None:
        try:
            st = os.stat(self.file_path)
            if st.st_size <= self.max_bytes:
                return
            # Читаем хвост последних max_bytes и перезаписываем файл
            with open(self.file_path, 'rb') as rf:
                if st.st_size > self.max_bytes:
                    rf.seek(st.st_size - self.max_bytes)
                tail = rf.read()
            with open(self.file_path, 'wb') as wf:
                wf.write(tail)
        except Exception:
            # Игнорируем ошибки тримминга, чтобы не ломать основной вывод
            pass


