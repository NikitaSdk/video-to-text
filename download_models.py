"""
Заранее скачивает все модели Whisper, которые использует main.py,
чтобы при распознавании ничего не нужно было качать.
Показывает прогресс в мегабайтах и процентах: по каждой модели и общий.

Запуск (один раз):
    python download_models.py

Модели сохраняются туда же, куда их качает main.py:
    C:\\Users\\<имя>\\.cache\\huggingface\\hub
Уже скачанные модели пропускаются, прерванная загрузка докачивается.
"""

import os
import sys
import time
import threading

# обычная HTTP-загрузка: файл растёт на диске, и прогресс можно честно измерить
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import snapshot_download, HfApi, constants          # noqa: E402
from huggingface_hub.utils import disable_progress_bars                  # noqa: E402
from faster_whisper.utils import _MODELS                                 # noqa: E402

disable_progress_bars()   # свои полоски вместо стандартных «Fetching 4 files»

MODELS = ["small", "medium", "large-v3-turbo", "large-v3"]
FILES = ["config.json", "preprocessor_config.json", "model.bin",
         "tokenizer.json", "vocabulary.*"]
MB = 1024 * 1024


def wanted(fname: str) -> bool:
    return fname in FILES or fname.startswith("vocabulary.")


def expected_size(repo: str) -> int:
    """Сколько байт весит модель (спрашиваем у сервера)."""
    try:
        info = HfApi().model_info(repo, files_metadata=True)
        return sum((s.size or 0) for s in info.siblings if wanted(s.rfilename))
    except Exception:
        return 0


def on_disk(repo: str) -> int:
    """Сколько байт этой модели уже лежит на диске (включая недокачанное)."""
    folder = os.path.join(constants.HF_HUB_CACHE,
                          "models--" + repo.replace("/", "--"), "blobs")
    total = 0
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            try:
                total += os.path.getsize(os.path.join(folder, f))
            except OSError:
                pass
    return total


def bar(frac: float, width: int = 30) -> str:
    filled = int(width * frac)
    return "█" * filled + "░" * (width - filled)


def fmt_time(sec: float) -> str:
    if sec <= 0 or sec > 99 * 3600:
        return "--:--"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main():
    print("Узнаю размеры моделей...")
    sizes = {n: expected_size(_MODELS[n]) for n in MODELS}
    grand_total = sum(sizes.values())
    print(f"Всего: {grand_total / MB / 1024:.1f} ГБ\n")

    ok, failed = [], []
    done_before = 0  # байты уже обработанных моделей (для общего прогресса)

    for i, name in enumerate(MODELS, 1):
        repo = _MODELS[name]
        total = sizes[name]
        print(f"[{i}/{len(MODELS)}] {name}  ({total / MB:.0f} МБ)")

        result = {}

        def worker():
            try:
                result["path"] = snapshot_download(repo, allow_patterns=FILES)
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        last_bytes, last_time, speed = on_disk(repo), time.time(), 0.0
        while t.is_alive():
            time.sleep(0.5)
            cur = min(on_disk(repo), total) if total else on_disk(repo)
            now = time.time()
            inst = (cur - last_bytes) / max(now - last_time, 1e-6)
            speed = inst if speed == 0 else speed * 0.8 + inst * 0.2   # сглаживание
            last_bytes, last_time = cur, now

            frac = cur / total if total else 0
            all_frac = (done_before + cur) / grand_total if grand_total else 0
            eta = (total - cur) / speed if speed > 0 else 0
            line = (f"\r  {bar(frac)} {frac * 100:5.1f}%  "
                    f"{cur / MB:6.0f}/{total / MB:.0f} МБ  "
                    f"{speed / MB:5.1f} МБ/с  осталось {fmt_time(eta)}  "
                    f"| всего {all_frac * 100:5.1f}%")
            sys.stdout.write(line)
            sys.stdout.flush()
        t.join()

        if "error" in result:
            print(f"\n  ОШИБКА: {result['error']}\n")
            failed.append(name)
        else:
            all_frac = (done_before + total) / grand_total if grand_total else 1
            sys.stdout.write(f"\r  {bar(1)} 100.0%  {total / MB:6.0f}/{total / MB:.0f} МБ"
                             f"  готово{' ' * 30}| всего {all_frac * 100:5.1f}%\n\n")
            ok.append(name)
        done_before += total

    print("=" * 60)
    print("Скачаны:", ", ".join(ok) or "—")
    if failed:
        print("Не удалось:", ", ".join(failed), "— запустите скрипт ещё раз, он докачает.")
    else:
        print("Все модели на месте. Теперь main.py не будет ничего качать.")


if __name__ == "__main__":
    main()