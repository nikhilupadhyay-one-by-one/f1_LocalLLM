# f1_LocalLLM
<!-- Alias SENTINEL -->

This project is dedicated to the LocalLLM implementation — a local LLM that runs like an actual AI, without needing an internet connection.

**Alias:** LLM

🏁 **FLAGSHIP PROJECT** 🏁

## Notes

1. This project only uses tools and models that support OpenVINO.
2. All supported models are listed in the `download_model_OpenVINO` folder — run `python {filename.py}` to install a given model.
3. The folder contains multiple models to choose from.

## Tech Stack

**Cloud Providers**
- Hugging Face — open-source model distributor
- Git — local commits, code recovery, and debugging
- GitHub — cloud provider for concurrent access
- Docker — seamless access across multiple platforms

**Languages**
- C++ — tray icon and auth shell
- Python — GUI, model handling, and inference
- SQL (SQLite) — local memory and database storage, referenced when needed

**Frameworks & Libraries**
- Qt6 Widgets — native system tray and password-window UI for the C++ shell
- PySide6 — Python GUI for the app's main mode windows, same Qt foundation as the C++ shell
- OpenVINO GenAI — AI inference engine, shared across both the C++ and Python halves; targets the local NPU/iGPU
- Piper — local text-to-speech (Phase 2)

**C++ ↔ Python Communication**
- Local TCP socket (`127.0.0.1`) — lightweight, cross-platform bridge between the auth shell and the Python app

**Build & Packaging**
- CMake — C++ build system
- Nuitka (primary) / PyInstaller (fallback) — bundling the Python half for end users without a Python install

**AI**
Honorable mentions to the AI tools that assisted in building this product:
- Claude — codebase and backend
- ChatGPT — codebase review and bug fixes
- Copilot — icon and wallpaper generation


**OPTIONS SUPPORTED**
1. Ideas
2. Music
3. Img Gen
4. ???
5. GO ONLINE!! (Requires your API key - The Memories of the model is restricted off limits to your online mode)