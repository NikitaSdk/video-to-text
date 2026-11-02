"""
Видео → текст.

Вставляете ссылку (YouTube или другой сайт) либо путь к файлу —
получаете txt со всем, что сказано в ролике.

Текст всегда берётся из звука. Субтитры сайта не используются.
У YouTube часто есть английский автодубляж: программа берёт оригинальную
дорожку, на языке которой реально говорят.

Режимы:
  Оригинал            — как говорят в видео
  Перевод на русский  — тот же текст по-русски
  Оригинал + перевод  — оба варианта

Запуск (из папки проекта, тем интерпретатором, где стоят библиотеки):
    python main.py
"""

import gc
import os
import re
import time
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor

LANGS = {
    "Авто": None, "Русский": "ru", "Украинский": "uk", "Английский": "en",
    "Немецкий": "de", "Французский": "fr", "Испанский": "es", "Итальянский": "it",
    "Португальский": "pt", "Польский": "pl", "Турецкий": "tr", "Арабский": "ar",
    "Китайский": "zh", "Японский": "ja", "Корейский": "ko",
}
AUTO = "Авто"
MODELS = [AUTO, "small", "medium", "large-v3-turbo", "large-v3"]
MODES = {"Оригинал": "orig", "Перевод на русский": "ru", "Оригинал + перевод": "both"}

CYRILLIC = re.compile("[А-Яа-яЁёІіЇїЄєҐґ]")
CYR_LANGS = ("ru", "uk", "be", "bg", "sr", "mk", "kk")
LANG_ALIAS = {"iw": "he", "jw": "jv", "in": "id"}

# Короткие фразы, которые Whisper дописывает сам, когда в звуке тишина.
HALLUCINATION_PARTS = (
    "субтитры сделал", "субтитры создал", "субтитры создавал", "субтитры подогнал",
    "dimatorzok", "продолжение следует", "thanks for watching", "thank you for watching",
    "подписывайтесь на канал", "amara.org",
)

PROMPTS = {
    "ru": "Лекция. Дословная расшифровка речи.",
    "uk": "Лекція. Дослівний запис мовлення.",
    "en": "Lecture. Verbatim transcript of the speech.",
    "de": "Vorlesung. Wortgetreues Transkript.",
    "fr": "Conférence. Transcription fidèle de la parole.",
    "es": "Conferencia. Transcripción literal.",
    "it": "Lezione. Trascrizione letterale.",
    "pt": "Aula. Transcrição literal.",
    "pl": "Wykład. Dosłowna transkrypcja.",
    "tr": "Ders. Birebir konuşma dökümü.",
    "ar": "محاضرة. تفريغ حرفي للكلام.",
}

# Порог ниже обычного и запас по краям, чтобы не срезать тихие слова.
VAD = dict(threshold=0.30, min_speech_duration_ms=0,
           min_silence_duration_ms=250, speech_pad_ms=450)

# Длинное видео режется кусками: меньше памяти, текст появляется по ходу.
# У куска есть запас после стыка, чтобы фраза не оборвалась. Следующий кусок
# начинается с конца последней уже записанной фразы, без повтора.
SLICE_SEC = 15 * 60
OVERLAP_SEC = 35
# До этой длины весь ролик считается одним куском.
ONE_SHOT_SEC = 20 * 60


def setup_cuda_dlls():
    """Подключает CUDA-библиотеки из pip, чтобы распознавание увидело видеокарту."""
    if os.name != "nt":
        return
    import glob
    import site
    bases = list(site.getsitepackages())
    try:
        bases.append(site.getusersitepackages())
    except Exception:
        pass
    for base in bases:
        for folder in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(folder)
            except Exception:
                pass
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")


setup_cuda_dlls()

CPU_THREADS = os.cpu_count() or 4
SAVE_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "Текстовые файл видео дорожек")
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()
_CUDA_BROKEN = False
_GOOGLE_BLOCKED = False
_ARGOS_INDEX_READY = False
_ARGOS_LOCK = threading.Lock()


def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    return url if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()) else None


def normalize_lang(code):
    if not code:
        return None
    code = str(code).split("-")[0].split("_")[0].lower()
    return LANG_ALIAS.get(code, code)


def lang_label(code):
    names = {value: name for name, value in LANGS.items() if value}
    if not code:
        return "не определён"
    name = names.get(code)
    return f"{name.lower()} ({code})" if name else code


def is_ru(code):
    return bool(code) and str(code).lower().startswith("ru")


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    return name[:120] or "transcript"


def fmt_dur(sec: float) -> str:
    sec = int(max(0, sec))
    hours, rest = divmod(sec, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def fmt_ts(sec: float, with_hours: bool) -> str:
    sec = int(max(0, sec))
    hours, rest = divmod(sec, 3600)
    minutes, seconds = divmod(rest, 60)
    if with_hours or hours:
        return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
    return f"[{minutes:02d}:{seconds:02d}]"


def looks_like_source(text: str) -> bool:
    text = text.strip().strip('"')
    if os.path.isfile(text):
        return True
    if re.match(r"https?://", text, re.I):
        return True
    return extract_video_id(text) is not None


def canonical_source(text: str) -> str:
    text = text.strip().strip('"')
    if os.path.isfile(text) or re.match(r"https?://", text, re.I):
        return text
    video_id = extract_video_id(text)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return text


def clean_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_hallucination(line: str) -> bool:
    folded = line.strip().lower()
    if len(folded) < 2:
        return True
    # Типичная выдумка Whisper в конце ролика, когда речь уже кончилась.
    if "редактор субтитров" in folded or "корректор а." in folded:
        return True
    if len(folded) > 90:
        return False
    if "спасибо за просмотр" in folded and len(folded) < 40:
        return True
    return any(part in folded for part in HALLUCINATION_PARTS)


def collapse_repeats(text: str) -> str:
    """Убирает фразу, повторённую подряд: так Whisper зацикливается на длинном звуке."""
    parts = re.split(r"(?<=[.!?…])\s+", text)
    kept = []
    for part in parts:
        if kept and part == kept[-1]:
            continue
        kept.append(part)
    return " ".join(kept)


# ---------- звук ----------

def _bitrate(fmt: dict) -> float:
    return float(fmt.get("abr") or fmt.get("tbr") or 0)


def _audio_only(fmt: dict) -> bool:
    if fmt.get("acodec") in (None, "none"):
        return False
    if fmt.get("vcodec") not in (None, "none"):
        return False
    return not fmt.get("has_drm")


def _is_original(fmt: dict) -> bool:
    note = (fmt.get("format_note") or "").lower()
    return "original" in note or (fmt.get("language_preference") or 0) >= 10


def choose_audio_format(info: dict):
    """Оригинальная речь, не английский автодубляж YouTube.

    Возвращает (format_id или None, подпись для журнала, код языка дорожки).
    Битрейт около 110 кбит/с: для речи этого достаточно, файл качается быстрее.
    Берём только дорожки, у которых уже есть прямая ссылка: остальные YouTube
    отдаёт без адреса, и скачивание обрывается.
    """
    audios = [fmt for fmt in (info.get("formats") or []) if _audio_only(fmt) and fmt.get("url")]
    if not audios:
        return None, "основной звук ролика", normalize_lang(info.get("language"))

    originals = [fmt for fmt in audios if _is_original(fmt)]
    if originals:
        pool, kind = originals, "оригинал"
    else:
        video_lang = normalize_lang(info.get("language"))
        same = [fmt for fmt in audios if normalize_lang(fmt.get("language")) == video_lang] if video_lang else []
        pool, kind = (same or audios), ("язык ролика" if same else "основная дорожка")

    def score(fmt):
        bitrate = _bitrate(fmt)
        if bitrate <= 0:
            distance = 50
        elif bitrate < 70:
            distance = (70 - bitrate) + 25
        else:
            distance = abs(bitrate - 110)
        codec = (fmt.get("acodec") or "") + (fmt.get("ext") or "")
        opus_penalty = 0 if ("opus" in codec or fmt.get("ext") == "webm") else 6
        return (distance, opus_penalty, -bitrate)

    best = min(pool, key=score)
    lang = normalize_lang(best.get("language")) or normalize_lang(info.get("language"))
    bitrate = _bitrate(best)
    parts = [lang_label(lang) if lang else "язык не указан", kind]
    if best.get("format_note"):
        parts.append(str(best["format_note"]))
    if bitrate:
        parts.append(f"{bitrate:.0f} кбит/с")
    return best.get("format_id"), ", ".join(parts), lang


def find_node():
    """Node нужен YouTube: без него часть роликов, включая Shorts, остаётся без звука."""
    import shutil
    found = shutil.which("node")
    if found and os.path.isfile(found):
        return found
    local = os.environ.get("LOCALAPPDATA", "")
    for path in (
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
        os.path.join(local, "Programs", "nodejs", "node.exe"),
    ):
        if path and os.path.isfile(path):
            return path
    return None


def ydl_common(**extra):
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        # Скрипт, который расшифровывает ссылки YouTube. Без него форматы пропадают.
        "remote_components": ["ejs:github"],
    }
    node = find_node()
    if node:
        opts["js_runtimes"] = {"node": {"path": node}}
    opts.update(extra)
    return opts


def explain_download_error(error) -> str:
    text = str(error or "")
    low = text.lower()
    if "sign in" in low or "not a bot" in low or "confirm you" in low:
        return ("YouTube просит подтвердить, что запрос не от робота. "
                "Откройте это видео в Chrome или Edge и нажмите «Получить текст» ещё раз.")
    if "private" in low or "removed" in low or "unavailable" in low or "not available" in low:
        return "Это видео недоступно: его удалили, закрыли или оно не открывается в вашем регионе."
    if "no video formats" in low or "requested format is not available" in low:
        return "YouTube не отдал звуковую дорожку. Попробуйте ещё раз или другую ссылку."
    if "reloaded" in low:
        return "YouTube временно отклонил запрос. Нажмите «Получить текст» ещё раз."
    if "timed out" in low or "connection" in low or "network" in low or "10054" in low:
        return "Нет связи с YouTube. Проверьте интернет и попробуйте снова."
    line = text.strip().splitlines()[-1] if text.strip() else "Не удалось скачать звук."
    line = re.sub(r"^ERROR:\s*", "", line).strip()
    return line[:400] or "Не удалось скачать звук."


def _has_audio_url(info: dict) -> bool:
    for fmt in info.get("formats") or []:
        if _audio_only(fmt) and fmt.get("url"):
            return True
    return bool(info.get("url"))


def _extract_info(url: str, extra: dict):
    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_common(skip_download=True, **extra)) as ydl:
        return ydl.extract_info(url, download=False)


def _probe_attempts():
    """Сначала обычный клиент YouTube, потом запасные, потом куки браузера."""
    attempts = [
        {},
        {"extractor_args": {"youtube": {"player_client": ["tv_embedded"]}}},
        {"extractor_args": {"youtube": {"player_client": ["web_safari"]}}},
        {"extractor_args": {"youtube": {"player_client": ["ios"]}}},
    ]
    for browser in ("edge", "chrome", "firefox", "brave"):
        attempts.append({"cookiesfrombrowser": (browser,)})
    return attempts


def _download_with(url: str, folder: str, format_spec: str, extra: dict):
    import yt_dlp
    opts = ydl_common(
        format=format_spec,
        outtmpl=os.path.join(folder, "audio.%(ext)s"),
        concurrent_fragment_downloads=8,
        **extra,
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    if not path or not os.path.exists(path) or os.path.getsize(path) < 1000:
        files = [os.path.join(folder, name) for name in os.listdir(folder)]
        files = [item for item in files if os.path.isfile(item) and os.path.getsize(item) >= 1000]
        if not files:
            raise RuntimeError("Не удалось скачать звук.")
        path = max(files, key=os.path.getsize)
    return path, info


def download_audio(url: str, folder: str, log):
    log("Смотрю звуковые дорожки...")
    info = None
    extra = {}
    last_error = None
    for index, attempt in enumerate(_probe_attempts()):
        try:
            got = _extract_info(url, attempt)
        except Exception as exc:
            # Ошибка «нет куки в браузере» не должна затирать настоящую причину с YouTube.
            if last_error is None or "cookie" not in str(exc).lower():
                last_error = exc
            if index == 0:
                log("Обычный запрос к YouTube не прошёл, пробую другой способ...")
            continue
        if _has_audio_url(got):
            info, extra = got, attempt
            if index:
                log("Нашёл звук другим способом.")
            break
        last_error = RuntimeError("YouTube не отдал звуковую дорожку.")
        if index == 0:
            log("В первом ответе нет звука, пробую другой способ...")
    if info is None:
        raise RuntimeError(explain_download_error(last_error))

    format_id, desc, track_lang = choose_audio_format(info)
    log(f"Беру дорожку: {desc}. Автодубляж не использую.")
    log("Скачиваю звук...")
    specs = []
    if format_id:
        specs.append(format_id)
    specs.append("ba[format_note*=original]/ba[language_preference>=10]/ba/b")
    specs.append("bestaudio/best")
    path = got = None
    last_error = None
    for spec in specs:
        try:
            path, got = _download_with(url, folder, spec, extra)
            break
        except Exception as exc:
            last_error = exc
            log("Эта дорожка не скачалась, пробую другую...")
    if not path:
        raise RuntimeError(explain_download_error(last_error))
    title = got.get("title") or info.get("title") or "video"
    duration = float(got.get("duration") or info.get("duration") or 0)
    track_lang = track_lang or normalize_lang(got.get("language")) or normalize_lang(info.get("language"))
    size_mb = os.path.getsize(path) / (1024 * 1024)
    log(f"Звук скачан ({size_mb:.1f} МБ) во временную папку. Видео не сохраняется, "
        "этот файл удалится сразу после распознавания.")
    return path, title, duration, track_lang


def probe_duration(path: str) -> float:
    import av
    with av.open(path, metadata_errors="ignore") as container:
        if container.duration:
            return float(container.duration) / av.time_base
        stream = container.streams.audio[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
    return 0.0


def read_clip(path: str, start: float, seconds: float, sampling_rate: int = 16000):
    """Кусок звука с нужного места, без чтения всего многочасового файла."""
    import av
    import numpy as np

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    chunks = []
    total = 0
    need = int(seconds * sampling_rate)
    start = max(0.0, start)
    with av.open(path, metadata_errors="ignore") as container:
        stream = container.streams.audio[0]
        if start > 0.05:
            # Смещение в единицах контейнера. Со stream= offset был бы в time_base
            # дорожки, и на WebM перемотка уезжала в конец файла.
            container.seek(int(start * av.time_base), backward=True, any_frame=False)
        frames = container.decode(stream)
        while True:
            try:
                frame = next(frames)
            except StopIteration:
                break
            except av.error.InvalidDataError:
                continue
            if frame.pts is not None and stream.time_base is not None:
                at = float(frame.pts * stream.time_base)
                if at < start - 0.05:
                    continue
            frame.pts = None
            for out in resampler.resample(frame):
                samples = np.asarray(out.to_ndarray()).reshape(-1).astype(np.float32)
                if samples.size and max(abs(float(samples.min())), abs(float(samples.max()))) > 1.5:
                    samples /= 32768.0
                chunks.append(samples)
                total += samples.size
                if total >= need:
                    break
            if total >= need:
                break
        if total < need:
            for out in resampler.resample(None):
                samples = np.asarray(out.to_ndarray()).reshape(-1).astype(np.float32)
                if samples.size and max(abs(float(samples.min())), abs(float(samples.max()))) > 1.5:
                    samples /= 32768.0
                chunks.append(samples)
                total += samples.size
    if not chunks:
        return np.zeros(sampling_rate, dtype=np.float32)
    audio = np.concatenate(chunks)
    return audio[:need]


# ---------- модель и распознавание ----------

def cuda_available() -> bool:
    if _CUDA_BROKEN:
        return False
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "failed to allocate" in text


def cuda_failed(error: BaseException) -> bool:
    text = str(error).lower()
    return "cublas" in text or "cudnn" in text or "cuda" in text


def get_model(name: str, device: str, compute: str, log):
    from faster_whisper import WhisperModel
    key = (name, device, compute)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        if device == "cuda":
            for old in list(_MODEL_CACHE):
                if old[1] == "cuda":
                    del _MODEL_CACHE[old]
            gc.collect()
        where = "видеокарту" if device == "cuda" else "процессор"
        log(f"Загружаю модель «{name}» на {where}...")
        kwargs = {"device": device, "compute_type": compute}
        if device == "cpu":
            kwargs["cpu_threads"] = CPU_THREADS
        _MODEL_CACHE[key] = WhisperModel(name, **kwargs)
        return _MODEL_CACHE[key]


def plan_model(device: str, duration: float, model_name: str, accurate: bool) -> dict:
    minutes = (duration or 0) / 60
    if model_name and model_name != AUTO:
        name, why = model_name, "выбрана вручную"
        if name == "large-v3" and minutes > 40:
            why += " — на длинной лекции она сильно медленнее turbo"
    elif device == "cuda":
        name = "large-v3-turbo"
        why = "точная и быстрая модель на видеокарте" if accurate else "самый быстрый режим на видеокарте"
    elif minutes > 90:
        name, why = "small", "длинный ролик на процессоре — иначе ждать часами"
    elif minutes > 25:
        name, why = ("medium" if accurate else "small"), "счёт на процессоре"
    else:
        name, why = ("large-v3-turbo" if accurate else "medium"), "короткий ролик на процессоре"

    if device == "cuda" and name == "large-v3":
        compute, beam, batch = "float16", 5, 4
    elif device == "cuda" and accurate:
        compute, beam, batch = "float16", 5, 8
    elif device == "cuda":
        compute, beam, batch = "int8_float16", 1, 16
    else:
        compute, beam, batch = "int8", (5 if accurate else 1), 4
    return {"name": name, "compute": compute, "beam": beam, "batch": batch, "why": why}


def _detect_clip(model, audio):
    lang, prob, probs = model.detect_language(audio, vad_filter=True)
    return lang, float(prob), dict(probs)


def resolve_language(model, path, duration, title, track_lang, user_lang, log):
    """Язык оригинальной дорожки важнее догадки по первым секундам.

    Whisper по умолчанию смотрит только начало. Если там тишина, музыка
    или (раньше) английский дубляж, русская лекция уезжала в английский текст.
    """
    if user_lang:
        log(f"Язык речи: {lang_label(user_lang)} — выбран вручную")
        return user_lang

    track_lang = normalize_lang(track_lang)
    if duration >= 90:
        points = [duration * 0.12, duration * 0.50, duration * 0.82]
    elif duration >= 45:
        points = [min(8.0, duration * 0.1), duration * 0.65]
    else:
        points = [0.0]

    votes = []
    for start in points:
        try:
            clip = read_clip(path, start, 24)
            if clip is None or len(clip) < 8000:
                continue
            votes.append(_detect_clip(model, clip))
        except Exception:
            continue

    detected, conf = None, 0.0
    if votes:
        score = {}
        for lang, prob, _probs in votes:
            score[lang] = score.get(lang, 0.0) + prob
        detected = max(score, key=score.get)
        same = [prob for lang, prob, _probs in votes if lang == detected]
        conf = sum(same) / len(same)

    if track_lang:
        support = 0.0
        if votes:
            support = sum(probs.get(track_lang, 0.0) for _lang, _prob, probs in votes) / len(votes)
        other = (
            detected and detected != track_lang and conf >= 0.92
            and all(lang == detected for lang, _prob, _probs in votes)
            and support < 0.04
        )
        if other:
            chosen, why = detected, f"по звуку {conf:.0%}"
        else:
            chosen = track_lang
            why = "язык оригинальной дорожки"
            if detected and detected != track_lang:
                why += f", звук давал {detected} ({conf:.0%})"
    elif detected:
        chosen, why = detected, f"по звуку, уверенность {conf:.0%}"
    else:
        log("Язык речи не определился — модель посмотрит начало ролика сама.")
        return None

    # Русское название при английской пометке: либо дубляж, либо лекция
    # на английском с русским заголовком. Решает середина записи, не название.
    if chosen == "en" and CYRILLIC.search(title or ""):
        try:
            start = duration * 0.35 if duration > 40 else 0.0
            clip = read_clip(path, start, 24)
            lang, prob, probs = _detect_clip(model, clip)
        except Exception:
            lang, prob, probs = None, 0.0, {}
        cyr = max(CYR_LANGS, key=lambda code: probs.get(code, 0.0)) if probs else "ru"
        if lang in CYR_LANGS and prob >= 0.35:
            chosen, why = lang, "в середине ролика эта речь, несмотря на английскую пометку"
        elif probs.get(cyr, 0.0) >= 0.15 and prob < 0.8:
            chosen, why = cyr, "название на кириллице, английский определён неуверенно"
        else:
            why = "речь английская, хотя название на русском"

    log(f"Язык речи: {lang_label(chosen)} — {why}")
    return chosen


class Paragraphs:
    def __init__(self):
        self.items = []
        self._buf = []
        self._start = None
        self._last_end = None
        self._prev = None

    def add(self, start: float, end: float, text: str):
        text = clean_line(text)
        if not text or is_hallucination(text):
            return None
        if text == self._prev:
            return None
        self._prev = text
        gap = 0.0 if self._last_end is None else start - self._last_end
        flushed = None
        if self._buf and (gap > 1.5 or sum(len(part) for part in self._buf) > 450):
            flushed = self.flush()
        if not self._buf:
            self._start = start
        self._buf.append(text)
        self._last_end = end
        return flushed

    def flush(self):
        if not self._buf:
            return None
        item = (self._start or 0.0, collapse_repeats(" ".join(self._buf).strip()))
        self.items.append(item)
        self._buf = []
        self._start = None
        return item


def word_count(paragraphs) -> int:
    return sum(len(text.split()) for _start, text in paragraphs)


def text_too_thin(paragraphs, duration: float) -> bool:
    """Речь и пение дают десятки слов в минуту. Горстка слов на длинный ролик — текст обрезан."""
    minimum = max(12, int((duration or 0) / 60 * 15))
    return word_count(paragraphs) < minimum


def run_batched(model, audio, lang, beam, batch, prompt, vad=None, clips=None):
    from faster_whisper import BatchedInferencePipeline
    pipe = BatchedInferencePipeline(model=model)
    kwargs = dict(
        language=lang,
        task="transcribe",
        beam_size=beam,
        batch_size=batch,
        without_timestamps=True,
        temperature=0.0,
        multilingual=False,
    )
    if clips:
        # Без фильтра тишины: иначе пение и тихая речь выкидываются целиком.
        kwargs["vad_filter"] = False
        kwargs["clip_timestamps"] = clips
    else:
        kwargs["vad_filter"] = True
        kwargs["vad_parameters"] = dict(vad or VAD)
    return pipe.transcribe(audio, **kwargs)


def _collect_segments(model, audio, lang, beam, batch, prompt, vad, offset, limit_rel, is_last):
    segments, info = run_batched(model, audio, lang, beam, batch, prompt, vad)
    rows = []
    for seg in segments:
        start = float(seg.start or 0)
        if not is_last and start >= limit_rel:
            continue
        text = seg.text or ""
        if text.strip():
            rows.append((offset + start, offset + float(seg.end or start), text))
    return rows, info


def transcribe_file(model, path, duration, lang, settings, log, on_paragraph):
    prompt = PROMPTS.get(lang)
    total = duration or probe_duration(path) or 1

    def once(vad, announce: bool):
        paragraphs = Paragraphs()
        started = time.time()
        last_pct = -1

        def commit(rows):
            nonlocal last_pct
            for start, end, text in rows:
                item = paragraphs.add(start, end, text)
                if item and on_paragraph:
                    on_paragraph(*item)
                pct = min(99, int(end / total * 100))
                if pct // 5 != last_pct // 5:
                    spent = time.time() - started
                    speed = end / spent if spent > 0 else 0
                    left = (total - end) / speed if speed > 1 else 0
                    log(f"  {pct}%   {fmt_dur(end)} из {fmt_dur(total)}"
                        f"   {speed:.0f}×   осталось ~{fmt_dur(left)}")
                    last_pct = pct

        def piece(audio, offset, limit_rel, is_last):
            batches = [settings["batch"]]
            if settings["batch"] > 8:
                batches += [8, 4]
            elif settings["batch"] > 4:
                batches.append(4)
            error = None
            for batch in batches:
                try:
                    rows, _info = _collect_segments(
                        model, audio, lang, settings["beam"], batch, prompt, vad,
                        offset, limit_rel, is_last)
                    settings["batch"] = batch
                    return rows
                except Exception as exc:
                    error = exc
                    if not is_oom(exc) or batch == batches[-1]:
                        raise
                    nxt = batches[batches.index(batch) + 1]
                    log(f"Видеокарте не хватило памяти, уменьшаю пакет до {nxt}...")
                    gc.collect()
            raise error

        if total <= ONE_SHOT_SEC:
            if announce:
                log("Распознаю речь... Текст сразу пишется в файл.")
            commit(piece(path, 0.0, 10 ** 9, True))
        else:
            approx = max(1, int((total + SLICE_SEC - 1) // SLICE_SEC))
            if announce:
                log(f"Распознаю речь частями примерно по 15 минут (~{approx} шт.). "
                    "Текст сразу пишется в файл.")
            position = 0.0
            index = 1
            while position < total - 0.3:
                remaining = total - position
                is_last = remaining <= SLICE_SEC + 1
                take = remaining if is_last else SLICE_SEC + OVERLAP_SEC
                log(f"Часть {index}: {fmt_dur(position)}–{fmt_dur(min(total, position + take))}")
                audio = read_clip(path, position, take)
                rows = piece(audio, position, 10 ** 9, True)
                if not is_last:
                    kept = []
                    for start, end, text in rows:
                        if start - position >= SLICE_SEC:
                            break
                        kept.append((start, end, text))
                    rows = kept
                commit(rows)
                if is_last:
                    break
                nxt = rows[-1][1] if rows else position + SLICE_SEC
                if nxt <= position + 0.5:
                    nxt = position + SLICE_SEC
                position = min(nxt, total)
                index += 1

        tail = paragraphs.flush()
        if tail and on_paragraph:
            on_paragraph(*tail)
        return paragraphs.items

    paragraphs = once(dict(VAD), True)
    if not text_too_thin(paragraphs, total):
        return paragraphs

    log(f"Получилось только {word_count(paragraphs)} слов на {fmt_dur(total)}. "
        "Фильтр тишины, скорее всего, выкинул пение или тихую речь. "
        "Распознаю запись целиком, без пропусков...")
    if on_paragraph:
        on_paragraph(None, None)
    paragraphs = Paragraphs()
    started = time.time()
    last_pct = -1
    segments, _info = model.transcribe(
        path,
        language=lang,
        task="transcribe",
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0.0,
        without_timestamps=False,
        no_speech_threshold=0.95,
    )
    for seg in segments:
        text = seg.text or ""
        if not text.strip():
            continue
        start = float(seg.start or 0)
        end = float(seg.end or start)
        item = paragraphs.add(start, end, text)
        if item and on_paragraph:
            on_paragraph(*item)
        pct = min(99, int(end / total * 100))
        if pct // 5 != last_pct // 5:
            spent = time.time() - started
            speed = end / spent if spent > 0 else 0
            left = (total - end) / speed if speed > 1 else 0
            log(f"  {pct}%   {fmt_dur(end)} из {fmt_dur(total)}"
                f"   {speed:.0f}×   осталось ~{fmt_dur(left)}")
            last_pct = pct
    tail = paragraphs.flush()
    if tail and on_paragraph:
        on_paragraph(*tail)
    log(f"Без пропусков получилось {word_count(paragraphs.items)} слов.")
    return paragraphs.items


def from_audio(url, lang, model_name, accurate, log, on_start=None, on_paragraph=None):
    """Возвращает (абзацы [(сек, текст)], язык, название, длительность)."""
    global _CUDA_BROKEN
    temporary = None
    try:
        if os.path.isfile(url):
            path, title, duration, track_lang = url, os.path.splitext(os.path.basename(url))[0], probe_duration(url), None
            log(f"Файл: {title}")
        else:
            temporary = tempfile.TemporaryDirectory()
            path, title, duration, track_lang = download_audio(url, temporary.name, log)
        if duration <= 0:
            duration = probe_duration(path)
        log(f"«{title}», длительность {fmt_dur(duration)}")
        if duration >= 45 * 60:
            log("Длинная лекция. Само распознавание на видеокарте обычно занимает несколько минут, "
                "скачивание зависит от интернета.")

        def recognize(device):
            settings = plan_model(device, duration, model_name, accurate)
            log(f"Модель: {settings['name']} — {settings['why']}")
            model = get_model(settings["name"], device, settings["compute"], log)
            language = resolve_language(model, path, duration, title, track_lang, lang, log)
            if on_start:
                on_start(title, language, duration)
            paragraphs = transcribe_file(
                model, path, duration, language, settings, log, on_paragraph)
            return paragraphs, language

        device = "cuda" if cuda_available() else "cpu"
        if device == "cpu":
            log("Видеокарта недоступна, считаю на процессоре. Длинная лекция займёт много времени.")
        try:
            paragraphs, language = recognize(device)
        except Exception as exc:
            if device != "cuda" or not cuda_failed(exc):
                raise
            _CUDA_BROKEN = True
            with _MODEL_LOCK:
                _MODEL_CACHE.clear()
            gc.collect()
            log(f"Видеокарта не запустилась ({exc}). Считаю на процессоре...")
            paragraphs, language = recognize("cpu")
        return paragraphs, language or "", title, duration
    finally:
        if temporary is not None:
            try:
                temporary.cleanup()
                log("Временный звук удалён. На диске остаётся только текстовый файл.")
            except Exception:
                log("Временный звук остался в папке Temp Windows. "
                    "Она чистится сама, видео программа не хранит.")


def preload_turbo():
    if not cuda_available():
        return
    try:
        get_model("large-v3-turbo", "cuda", "float16", lambda _message: None)
    except Exception:
        pass


# ---------- перевод ----------

def split_chunks(text: str, limit: int) -> list:
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit and current:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


def google_code(code):
    code = normalize_lang(code) or "auto"
    return {"zh": "zh-CN", "he": "iw"}.get(code, code)


def _google_one(text: str, source: str) -> str:
    from deep_translator import GoogleTranslator
    translated = GoogleTranslator(source=source, target="ru").translate(text)
    return (translated or "").strip()


def translate_google(text: str, source: str, log) -> str:
    source = google_code(source)
    chunks = split_chunks(text, 4000)
    done = [0]
    lock = threading.Lock()

    def work(chunk: str) -> str:
        delay = 1.0
        last = None
        for _attempt in range(4):
            try:
                result = _google_one(chunk, source)
                with lock:
                    done[0] += 1
                    log(f"  перевод {done[0]}/{len(chunks)}")
                return result
            except Exception as exc:
                last = exc
                time.sleep(delay)
                delay = min(delay * 2, 8)
        raise last

    with ThreadPoolExecutor(max_workers=3) as pool:
        return "\n".join(pool.map(work, chunks))


def translate_offline(text: str, source: str, log) -> str:
    global _ARGOS_INDEX_READY
    import argostranslate.package as package
    import argostranslate.translate as translator

    source = normalize_lang(source) or "en"
    target = "ru"

    def ensure(src: str, dst: str) -> bool:
        global _ARGOS_INDEX_READY
        with _ARGOS_LOCK:
            installed = package.get_installed_packages()
            if any(item.from_code == src and item.to_code == dst for item in installed):
                return True
            if not _ARGOS_INDEX_READY:
                package.update_package_index()
                _ARGOS_INDEX_READY = True
            log(f"  скачиваю офлайн-словарь {src}→{dst} (один раз)...")
            return bool(package.install_package_for_language_pair(src, dst))

    if ensure(source, target):
        steps = [(source, target)]
    elif source != "en" and ensure(source, "en") and ensure("en", target):
        steps = [(source, "en"), ("en", target)]
    else:
        raise RuntimeError(f"нет офлайн-перевода для языка «{source}»")

    chunks = split_chunks(text, 2000)
    output = []
    for index, chunk in enumerate(chunks, 1):
        result = chunk
        for src, dst in steps:
            result = translator.translate(result, src, dst)
        output.append((result or "").strip())
        log(f"  офлайн-перевод {index}/{len(chunks)}")
    return "\n".join(output)


def translate_paragraphs(paragraphs, source, log):
    global _GOOGLE_BLOCKED
    texts = [text for _start, text in paragraphs]
    body = "\n@@\n".join(texts)
    if not _GOOGLE_BLOCKED:
        try:
            log("Перевожу на русский через Google Переводчик...")
            translated = translate_google(body, source, log)
        except Exception as exc:
            _GOOGLE_BLOCKED = True
            log(f"Google временно недоступен ({type(exc).__name__}). Перевожу на компьютере...")
            translated = translate_offline(body, source, log)
    else:
        log("Перевожу на русский на компьютере (Google сейчас не отвечает)...")
        translated = translate_offline(body, source, log)
    parts = [part.strip() for part in translated.split("@@")]
    parts = [part for part in parts if part]
    if len(parts) == len(texts):
        return [(start, part) for (start, _text), part in zip(paragraphs, parts)]
    # Переводчик склеил абзацы — метки времени остаются, текст не теряется.
    merged = re.sub(r"\s*@@\s*", "\n", translated).strip() or "\n".join(texts)
    return [(paragraphs[0][0], merged)] if paragraphs else []


def render_document(title, url, paragraphs, duration, language, mode, translated=None) -> str:
    hours = (duration or 0) >= 3600
    lines = [
        title,
        url,
        f"Длительность: {fmt_dur(duration or 0)}",
        f"Язык: {lang_label(language)}",
        "",
    ]

    def block(items):
        for start, text in items:
            lines.append(fmt_ts(start, hours))
            lines.append(text)
            lines.append("")

    if mode == "ru" and translated is not None:
        block(translated)
    elif mode == "both" and translated is not None:
        lines.append(f"=== ОРИГИНАЛ ({language or '?'}) ===")
        lines.append("")
        block(paragraphs)
        lines.append("=== ПЕРЕВОД НА РУССКИЙ ===")
        lines.append("")
        block(translated)
    else:
        block(paragraphs)
    return "\n".join(lines).rstrip() + "\n"


# ---------- окно ----------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Видео → текст")
        self.geometry("700x520")
        self.minsize(560, 440)
        icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        if os.path.isfile(icon):
            try:
                self.iconbitmap(icon)
            except tk.TclError:
                pass

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Label(frame, text="Ссылка на видео или файл:").pack(anchor="w")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(2, 8))
        self.url = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self.url)
        entry.pack(side="left", fill="x", expand=True)
        entry.focus()
        ttk.Button(row, text="Вставить", command=self.paste).pack(side="left", padx=(6, 0))

        options = ttk.Frame(frame)
        options.pack(fill="x")
        ttk.Label(options, text="Язык видео:").pack(side="left")
        self.lang = tk.StringVar(value="Авто")
        ttk.Combobox(options, textvariable=self.lang, values=list(LANGS),
                     width=16, state="readonly").pack(side="left", padx=(4, 14))
        ttk.Label(options, text="Модель:").pack(side="left")
        self.model = tk.StringVar(value=AUTO)
        ttk.Combobox(options, textvariable=self.model, values=MODELS,
                     width=16, state="readonly").pack(side="left", padx=4)

        mode_row = ttk.Frame(frame)
        mode_row.pack(fill="x", pady=(8, 0))
        ttk.Label(mode_row, text="Текст:").pack(side="left")
        self.mode = tk.StringVar(value="Оригинал")
        for name in MODES:
            ttk.Radiobutton(mode_row, text=name, value=name,
                            variable=self.mode).pack(side="left", padx=(8, 0))

        checks = ttk.Frame(frame)
        checks.pack(fill="x", pady=6)
        self.accurate = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            checks,
            text="Максимальная точность (чуть медленнее)",
            variable=self.accurate,
        ).pack(side="left")

        output = ttk.Frame(frame)
        output.pack(fill="x")
        ttk.Label(output, text="Сохранять в:").pack(side="left")
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.out_dir = tk.StringVar(value=SAVE_DIR)
        ttk.Entry(output, textvariable=self.out_dir).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(output, text="...", width=3, command=self.choose_dir).pack(side="left")

        self.btn = ttk.Button(frame, text="Получить текст", command=self.start)
        self.btn.pack(pady=10)

        self.logbox = tk.Text(frame, height=12, state="disabled", wrap="word")
        self.logbox.pack(fill="both", expand=True)

        self.bind("<Return>", lambda _event: self.start())
        self.bind_all("<Control-KeyPress>", self.on_ctrl_key)

        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Вставить", command=lambda: self.menu_widget.event_generate("<<Paste>>"))
        self.menu.add_command(label="Копировать", command=lambda: self.menu_widget.event_generate("<<Copy>>"))
        self.menu.add_command(label="Вырезать", command=lambda: self.menu_widget.event_generate("<<Cut>>"))
        self.menu_widget = entry
        for widget in (entry, self.logbox):
            widget.bind("<Button-3>", self.show_menu)

        threading.Thread(target=preload_turbo, daemon=True).start()

    def on_ctrl_key(self, event):
        actions = {86: "<<Paste>>", 67: "<<Copy>>", 88: "<<Cut>>", 65: "<<SelectAll>>"}
        if event.keysym.lower() in ("v", "c", "x", "a"):
            return None
        action = actions.get(event.keycode)
        if action:
            event.widget.event_generate(action)
            return "break"
        return None

    def show_menu(self, event):
        self.menu_widget = event.widget
        self.menu.tk_popup(event.x_root, event.y_root)

    def paste(self):
        try:
            self.url.set(self.clipboard_get().strip())
        except tk.TclError:
            pass

    def choose_dir(self):
        folder = filedialog.askdirectory(initialdir=self.out_dir.get())
        if folder:
            self.out_dir.set(folder)

    def log(self, message: str):
        def write():
            self.logbox.configure(state="normal")
            self.logbox.insert("end", message + "\n")
            self.logbox.see("end")
            self.logbox.configure(state="disabled")
        self.after(0, write)

    def start(self):
        url = canonical_source(self.url.get())
        if not looks_like_source(url):
            messagebox.showerror("Ошибка", "Нужна ссылка на видео или путь к файлу.")
            return
        self.url.set(url)
        self.btn.configure(state="disabled")
        self.logbox.configure(state="normal")
        self.logbox.delete("1.0", "end")
        self.logbox.configure(state="disabled")
        threading.Thread(target=self.work, args=(url,), daemon=True).start()

    def work(self, url: str):
        started = time.time()
        lang = LANGS[self.lang.get()]
        mode = MODES[self.mode.get()]
        suffix = {"orig": "", "ru": " (RU)", "both": " (оригинал + RU)"}[mode]
        live = {"path": None, "header": ""}

        def on_start(title, language, duration):
            os.makedirs(self.out_dir.get(), exist_ok=True)
            path = os.path.join(self.out_dir.get(), safe_filename(title) + suffix + ".txt")
            live["path"] = path
            note = ""
            if mode != "orig":
                note = "[идёт распознавание — перевод будет в итоговом файле]\n\n"
            live["header"] = (f"{title}\n{url}\nДлительность: {fmt_dur(duration)}\n"
                              f"Язык: {lang_label(language)}\n\n{note}")
            with open(path, "w", encoding="utf-8") as file:
                file.write(live["header"])
            self.log(f"Файл создан, текст пишется по ходу: {path}")

        def on_paragraph(start, text):
            path = live["path"]
            if not path:
                return
            if start is None:
                with open(path, "w", encoding="utf-8") as file:
                    file.write(live["header"])
                return
            with open(path, "a", encoding="utf-8") as file:
                file.write(f"{fmt_ts(start, False)}\n{text}\n\n")

        try:
            paragraphs, src_lang, title, duration = from_audio(
                url, lang, self.model.get(), self.accurate.get(), self.log,
                on_start, on_paragraph)
            if not paragraphs or not any(text.strip() for _start, text in paragraphs):
                raise RuntimeError(
                    "В видео не удалось распознать речь. "
                    "Выберите язык вручную и модель large-v3.")

            translated = None
            if mode != "orig":
                if is_ru(src_lang):
                    translated = paragraphs
                    self.log("Речь уже на русском, отдельный перевод не нужен.")
                else:
                    try:
                        translated = translate_paragraphs(paragraphs, src_lang, self.log)
                    except Exception as exc:
                        self.log(f"Перевод не удался ({exc}). Сохраняю оригинал.")
                        translated = [(paragraphs[0][0],
                                       "[Перевод не удался. Ниже — оригинальный текст.]\n\n"
                                       + "\n".join(text for _s, text in paragraphs))]

            path = live["path"] or os.path.join(
                self.out_dir.get(), safe_filename(title) + suffix + ".txt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            document = render_document(title, url, paragraphs, duration, src_lang, mode, translated)
            with open(path, "w", encoding="utf-8") as file:
                file.write(document)
            spent = time.time() - started
            speed = (duration / spent) if spent > 0 else 0
            self.log(f"Готово за {fmt_dur(spent)} ({speed:.0f}× быстрее ролика). Файл: {path}")
            self.after(0, lambda: messagebox.showinfo("Готово", f"Текст сохранён:\n{path}"))
        except Exception as exc:
            self.log(f"Ошибка: {exc}")
            if live["path"]:
                self.log(f"Всё, что успело распознаться, сохранено в: {live['path']}")
            self.after(0, lambda message=str(exc): messagebox.showerror("Ошибка", message))
        finally:
            self.after(0, lambda: self.btn.configure(state="normal"))


if __name__ == "__main__":
    App().mainloop()
