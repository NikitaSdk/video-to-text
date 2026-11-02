# Video to Text

**Desktop app that turns any video into a text file.** Paste a link (YouTube or another site) or pick a file — get a `.txt` with everything that is said.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![faster-whisper](https://img.shields.io/badge/faster--whisper-76B900)
![yt-dlp](https://img.shields.io/badge/yt--dlp-FF0000?logo=youtube&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-3776AB)

## Features

- Text always comes **from the audio**, not from site subtitles.
- Picks the **original audio track** when YouTube offers an auto-dubbed one.
- **15 languages** with auto-detection; modes: original, Russian translation, or both.
- Removes typical Whisper **hallucinations** on silence ("thanks for watching", etc.).
- `download_models.py` pre-downloads all Whisper models with progress in MB and %.
- Packaged as a Windows `.exe` with PyInstaller (`build_app.bat`).

## Run

```bash
pip install faster-whisper yt-dlp deep-translator argostranslate
python download_models.py   # once
python main.py
```
