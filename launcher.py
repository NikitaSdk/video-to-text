"""Запуск программы двойным щелчком. Собирается в exe."""
import os
import subprocess
import sys

PROJECT = r"C:\Users\user\PycharmProjects\MP4-TXT"
CREATE_NO_WINDOW = 0x08000000


def project_dir():
    if os.path.isfile(os.path.join(PROJECT, "main.py")):
        return PROJECT
    here = os.path.dirname(os.path.abspath(sys.executable))
    if os.path.isfile(os.path.join(here, "main.py")):
        return here
    return PROJECT


def show_error(text):
    import ctypes
    ctypes.windll.user32.MessageBoxW(0, text, "Видео в текст", 0x10)


def main():
    root = project_dir()
    python = os.path.join(root, ".venv", "Scripts", "pythonw.exe")
    script = os.path.join(root, "main.py")
    if not os.path.isfile(python) or not os.path.isfile(script):
        show_error("Не найдены файлы программы в папке:\n" + root)
        return
    subprocess.Popen([python, script], cwd=root, creationflags=CREATE_NO_WINDOW)


if __name__ == "__main__":
    main()
