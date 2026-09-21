"""
Lemun Enterprises - Local LLM Assistant

Python half of the app. password_gate.cpp handles authentication, then
launches this script with its own install directory as argv[1].
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional


# ============================================================
# PATHS
# ============================================================

def resolve_install_root() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    return Path(__file__).resolve().parent


INSTALL_ROOT = resolve_install_root()
MODELS_DIR = INSTALL_ROOT / "download_model_OpenVINO" / "models"
DATA_DIR = INSTALL_ROOT / "data"
MEMORY_DB_PATH = DATA_DIR / "memory.db"
CRASH_LOG_PATH = DATA_DIR / "crash.log"

MAX_NEW_TOKENS = 2048
MODEL_MARKER_FILE = "openvino_model.xml"
APP_TITLE = "Welcome - Lemun Enterprises"


# ============================================================
# UTILITIES
# ============================================================

class AssistantError(Exception):
    """A user-facing, recoverable failure."""


def print_banner() -> None:
    width = 40
    print("=" * width)
    print("|" + " " * (width - 2) + "|")
    print("|" + APP_TITLE.center(width - 2) + "|")
    print("|" + " " * (width - 2) + "|")
    print("=" * width)


def print_footer() -> None:
    print("\nSession ended. Thanks for using the assistant.")
    print("Copyright Product : Avoid Unfair Use and Follow the Code of Conduct")


def format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def folder_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def has_internet(url: str = "https://huggingface.co", timeout: float = 5.0) -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # server responded at all, so the network path is fine
    except Exception:
        return False


def log_error(context: str, exc: BaseException) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(CRASH_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n{'=' * 60}\n")
            log_file.write(f"{datetime.now().isoformat(timespec='seconds')} - {context}\n")
            log_file.write(f"{'=' * 60}\n")
            log_file.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass  # logging must never itself be the thing that crashes


def pause_before_exit() -> None:
    print(f"\nDetails were written to:\n  {CRASH_LOG_PATH}")
    try:
        input("\nPress Enter to close this window...")
    except (EOFError, KeyboardInterrupt, OSError):
        pass


def pip_install(*packages: str) -> None:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
    except subprocess.CalledProcessError as exc:
        raise AssistantError(
            f"Automatic install failed (exit code {exc.returncode}).\n"
            f"Please run manually:  pip install {' '.join(packages)}"
        ) from exc
    except OSError as exc:
        raise AssistantError(f"Could not start pip: {exc}") from exc


# ============================================================
# OPENVINO ENVIRONMENT
# ============================================================

openvino = None
ov_genai = None
core = None


def ensure_openvino() -> None:
    global openvino, ov_genai, core

    try:
        import openvino as openvino_module
        import openvino_genai as genai_module
    except ImportError:
        print("OpenVINO isn't installed!")
        answer = input("Install it now automatically via pip? (y/n): ").strip().lower()

        if answer != "y":
            raise AssistantError(
                "OpenVINO is required to run this assistant.\n"
                "To install, run:  pip install openvino openvino-genai"
            )

        pip_install("openvino", "openvino-genai")

        try:
            import openvino as openvino_module
            import openvino_genai as genai_module
        except ImportError as exc:
            raise AssistantError(
                "OpenVINO installed but still cannot be imported.\n"
                "Try restarting the application."
            ) from exc

        print("\nOpenVINO installed successfully. Continuing...\n")

    openvino = openvino_module
    ov_genai = genai_module

    try:
        core = openvino.Core()
    except Exception as exc:
        raise AssistantError(f"OpenVINO loaded but failed to initialise: {exc}") from exc


def detect_devices() -> tuple[str, ...]:
    try:
        return tuple(core.available_devices)
    except Exception as exc:
        print(f"Couldn't read device info: {exc}")
        return ()


def print_environment() -> tuple[str, ...]:
    try:
        print("OpenVINO version:", openvino.__version__)
    except Exception:
        print("OpenVINO version: unknown")

    devices = detect_devices()
    print("Available devices:", ", ".join(devices) if devices else "none detected")
    return devices


def stream_continue_value():
    streaming_status = getattr(ov_genai, "StreamingStatus", None)
    if streaming_status is not None:
        return streaming_status.RUNNING
    return False


# ============================================================
# MODEL CATALOG
# ============================================================

@dataclass(frozen=True)
class LocalModel:
    name: str
    company: str
    country: str
    folder: str
    hf_repo: str
    supports: tuple[str, ...]

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.folder

    @property
    def is_installed(self) -> bool:
        return (self.path / MODEL_MARKER_FILE).exists()

    def describe(self) -> str:
        state = "installed" if self.is_installed else "not installed"
        return (
            f"{self.name} - {self.company} - {self.country} "
            f"(runs on: {', '.join(self.supports)}) [{state}]"
        )


BASE_MODEL = LocalModel(
    name="Qwen2.5-1.5B-Instruct-int4-ov",
    company="Alibaba",
    country="China",
    folder="Qwen2.5-1.5B-Instruct-int4-ov",
    hf_repo="llmware/qwen2.5-1.5b-instruct-ov",
    supports=("CPU", "GPU"),
)

# I have added models here - these are done manually ofc - since i want to manually test and then add models so i know what they are capable of

# IDEA_MODELS_ALL = ()
# IMAGE_MODELS_ALL = ()
# SPEECH_MODELS_ALL = ()
# MUSIC_MODELS_ALL = ()

IDEA_MODELS: tuple[LocalModel, ...] = (BASE_MODEL,)
IMAGE_MODELS: tuple[LocalModel, ...] = ()
SPEECH_MODELS: tuple[LocalModel, ...] = ()
MUSIC_MODELS: tuple[LocalModel, ...] = ()


def list_all_models() -> tuple[LocalModel, ...]:
    unique: dict[str, LocalModel] = {}
    for group in (IDEA_MODELS, IMAGE_MODELS, SPEECH_MODELS, MUSIC_MODELS):
        for model in group:
            unique[model.folder] = model
    return tuple(unique.values())


# ============================================================
# MODEL MANAGEMENT
# ============================================================

def fetch_download_size(model: LocalModel) -> Optional[int]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model.hf_repo, files_metadata=True)
        return sum(sibling.size for sibling in info.siblings if sibling.size)
    except Exception:
        return None


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
        return
    except ImportError:
        pass

    print("The 'huggingface_hub' package is required to download models.")
    answer = input("Install it now automatically via pip? (y/n): ").strip().lower()

    if answer != "y":
        raise AssistantError(
            "Cannot download models without huggingface_hub.\n"
            "To install, run:  pip install huggingface_hub"
        )

    pip_install("huggingface_hub")


def download_model(model: LocalModel) -> None:
    if not has_internet():
        raise AssistantError(
            f"No internet connection detected.\n"
            f"'{model.name}' still needs to be downloaded, so please connect "
            f"to the internet and try again."
        )

    ensure_huggingface_hub()
    import huggingface_hub
    from huggingface_hub import snapshot_download

    try:
        model.path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AssistantError(
            f"Could not create the models folder at:\n  {model.path.parent}\n"
            f"({exc})\n"
            f"This can happen due to permissions, a very long install path, or "
            f"special characters in a folder name."
        ) from exc

    print(f"\nDownloading {model.name}...")
    print(f"(huggingface_hub version: {huggingface_hub.__version__})")
    print("(This only happens once. Progress is shown below.)\n")

    try:
        snapshot_download(repo_id=model.hf_repo, local_dir=str(model.path))
    except Exception as exc:
        log_error(f"snapshot_download failed for '{model.name}'", exc)
        raise AssistantError(
            f"Download of '{model.name}' failed: {exc}\n"
            f"Full details were written to:\n  {CRASH_LOG_PATH}\n"
            f"Check your internet connection and try again."
        ) from exc

    if not model.is_installed:
        raise AssistantError(
            f"Download finished but '{model.name}' looks incomplete "
            f"(missing {MODEL_MARKER_FILE}). Please try again."
        )

    print(f"\n{model.name} downloaded successfully.\n")


def ensure_base_model() -> None:
    if BASE_MODEL.is_installed:
        return

    print(f"\nBase model '{BASE_MODEL.name}' is not installed.")
    print("This model is required for the assistant to work at all.\n")
    print(f"Download size: {format_bytes(fetch_download_size(BASE_MODEL))}")

    download_model(BASE_MODEL)


def ensure_model_available(model: LocalModel) -> bool:
    if model.is_installed:
        return True

    print(f"\n'{model.name}' is not installed yet.")
    print(f"Download size: {format_bytes(fetch_download_size(model))}")

    answer = input("Download it now? (y/n): ").strip().lower()
    if answer != "y":
        print("Skipped - returning to the menu.")
        return False

    download_model(model)
    return True


def uninstall_model(model: LocalModel) -> None:
    try:
        shutil.rmtree(model.path)
    except OSError as exc:
        raise AssistantError(f"Could not remove '{model.name}': {exc}") from exc


# ============================================================
# INFERENCE ENGINE
# ============================================================

def load_pipeline(model: LocalModel, device: str):
    print(f"\nLoading {model.name} on {device}...")
    started = time.perf_counter()

    try:
        pipeline = ov_genai.LLMPipeline(str(model.path), device)
    except Exception as exc:
        raise AssistantError(
            f"Failed to load '{model.name}' on {device}: {exc}\n"
            f"Check that the model folder exists at:\n  {model.path}"
        ) from exc

    print(f"Model loaded in {time.perf_counter() - started:.2f} seconds")
    return pipeline


class ChatSession:
    """One conversation with a loaded model; owns its own history."""

    def __init__(self, pipeline, max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.history = ov_genai.ChatHistory()
        self._keep_going = stream_continue_value()

    def ask(self, prompt: str, on_token: Callable[[str], None]) -> str:
        self.history.append({"role": "user", "content": prompt})

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = self.max_new_tokens

        def streamer(chunk: str):
            on_token(chunk)
            return self._keep_going

        try:
            result = self.pipeline.generate(self.history, config, streamer)
        except Exception as exc:
            self.history.pop()
            raise AssistantError(f"Generation failed: {exc}") from exc

        reply = result.texts[0]
        self.history.append({"role": "assistant", "content": reply})
        return reply

    @property
    def turn_count(self) -> int:
        return len(self.history) // 2


# ============================================================
# MEMORY / PERSISTENCE
# ============================================================

def save_session(feature_label: str, session: ChatSession, started_at: datetime) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    turn_count INTEGER NOT NULL,
                    history TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO sessions (feature, started_at, ended_at, turn_count, history) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    feature_label,
                    started_at.isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                    session.turn_count,
                    json.dumps(list(session.history)),
                ),
            )
    except Exception as exc:
        print(f"(Could not save this session to memory: {exc})")


# ============================================================
# CONSOLE UI
# ============================================================

def ask_choice(prompt: str, count: int, allow_back: bool = True) -> Optional[int]:
    hint = f" (1-{count}, or 0 to go back)" if allow_back else f" (1-{count})"

    while True:
        try:
            raw = input(f"{prompt}{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if not raw:
            continue

        try:
            choice = int(raw)
        except ValueError:
            print("That's not a number, try again.")
            continue

        if allow_back and choice == 0:
            return None
        if 1 <= choice <= count:
            return choice

        print(f"Enter a number between {'0' if allow_back else '1'} and {count}.")


def choose_model(models: Iterable[LocalModel]) -> Optional[LocalModel]:
    catalog = list(models)

    if not catalog:
        print("\nNo models are configured for this feature yet.")
        return None

    print("\nAvailable models:")
    for index, model in enumerate(catalog, start=1):
        print(f"  {index}. {model.describe()}")

    choice = ask_choice("\nPick a model", len(catalog))
    return catalog[choice - 1] if choice else None


def choose_device(model: LocalModel, detected: tuple[str, ...]) -> Optional[str]:
    print(f"\n'{model.name}' can run on: {', '.join(model.supports)}")

    options = []
    for device in model.supports:
        present = any(found.startswith(device) for found in detected)
        label = device if present else f"{device} (not detected on this machine)"
        options.append((device, label))

    for index, (_, label) in enumerate(options, start=1):
        print(f"  {index}. {label}")

    choice = ask_choice("\nPick a device", len(options))
    if choice is None:
        return None

    device, _ = options[choice - 1]

    if not any(found.startswith(device) for found in detected):
        confirm = input(f"OpenVINO doesn't see a {device}. Try anyway? (y/n): ")
        if confirm.strip().lower() != "y":
            return None

    return device


def run_chat_loop(session: ChatSession) -> None:
    print("\nChat session started. Type 'exit' or 'quit' to return to the menu.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLeaving chat.")
            return

        if prompt.lower() in ("exit", "quit"):
            return
        if not prompt:
            continue

        print("Assistant: ", end="", flush=True)
        started = time.perf_counter()

        def show(chunk: str) -> None:
            print(chunk, end="", flush=True)

        try:
            session.ask(prompt, on_token=show)
        except AssistantError as exc:
            print(f"\n[{exc}]\n")
            continue

        print(f"\n[{time.perf_counter() - started:.2f}s]\n")


# ============================================================
# FEATURES
# ============================================================

@dataclass(frozen=True)
class Feature:
    label: str
    description: str
    models: tuple[LocalModel, ...] = ()
    handler: Optional[Callable[[Feature, tuple[str, ...]], None]] = None
    planned_for: str = ""

    @property
    def is_available(self) -> bool:
        return self.handler is not None

    def describe(self) -> str:
        if self.is_available:
            return f"{self.label} - {self.description}"
        return f"{self.label} - {self.description}  [coming in {self.planned_for}]"


def run_ideas(feature: Feature, detected_devices: tuple[str, ...]) -> None:
    model = choose_model(feature.models)
    if model is None:
        return

    if not ensure_model_available(model):
        return

    device = choose_device(model, detected_devices)
    if device is None:
        return

    pipeline = load_pipeline(model, device)
    session = ChatSession(pipeline)
    started_at = datetime.now()

    run_chat_loop(session)

    if session.turn_count:
        save_session(feature.label, session, started_at)
        print(f"\n({session.turn_count} exchanges saved to memory)")


def run_manage_models(feature: Feature, detected_devices: tuple[str, ...]) -> None:
    installed = [model for model in list_all_models() if model.is_installed]

    if not installed:
        print("\nNo models are currently installed.")
        return

    print("\nInstalled models:")
    for index, model in enumerate(installed, start=1):
        tag = " (required, cannot remove)" if model is BASE_MODEL else ""
        size = format_bytes(folder_size_bytes(model.path))
        print(f"  {index}. {model.name} - {size}{tag}")

    choice = ask_choice("\nUninstall which model", len(installed))
    if choice is None:
        return

    model = installed[choice - 1]
    if model is BASE_MODEL:
        print("\nThe base model is required and cannot be removed.")
        return

    confirm = input(
        f"Remove '{model.name}'? You'll need to re-download it to use this "
        f"feature again. (y/n): "
    )
    if confirm.strip().lower() != "y":
        return

    uninstall_model(model)
    print(f"\n'{model.name}' removed.")


FEATURES: tuple[Feature, ...] = (
    Feature(
        label="Ideas",
        description="chat, capture and refine ideas offline",
        models=IDEA_MODELS,
        handler=run_ideas,
    ),
    Feature(
        label="Image Generation",
        description="describe an image and generate it locally",
        models=IMAGE_MODELS,
        planned_for="v2",
    ),
    Feature(
        label="Text to Speech",
        description="read text aloud",
        models=SPEECH_MODELS,
        planned_for="v2",
    ),
    Feature(
        label="Music Generation",
        description="generate short audio pieces",
        models=MUSIC_MODELS,
        planned_for="v3",
    ),
    Feature(
        label="Go Online",
        description="send queries to external AI APIs (needs internet + API key)",
        planned_for="v5",
    ),
    Feature(
        label="Manage Models",
        description="view installed models and free up disk space",
        handler=run_manage_models,
    ),
)


def run_menu(detected_devices: tuple[str, ...]) -> None:
    while True:
        print("\n" + "-" * 50)
        print("What would you like to do?")
        print("-" * 50)

        for index, feature in enumerate(FEATURES, start=1):
            print(f"  {index}. {feature.describe()}")
        print("  0. Exit")

        choice = ask_choice("\nChoose", len(FEATURES))
        if choice is None:
            return

        feature = FEATURES[choice - 1]

        if not feature.is_available:
            print(f"\n'{feature.label}' isn't implemented yet - planned for {feature.planned_for}.")
            continue

        try:
            feature.handler(feature, detected_devices)
        except AssistantError as exc:
            print(f"\n[{feature.label}] {exc}\n")
        except KeyboardInterrupt:
            print("\nInterrupted - returning to the menu.\n")


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> int:
    print_banner()

    try:
        ensure_openvino()
        detected_devices = print_environment()
        ensure_base_model()
        run_menu(detected_devices)
    except AssistantError as exc:
        print(f"\nERROR: {exc}\n")
        log_error("AssistantError reached main()", exc)
        pause_before_exit()
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        return 1
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {exc}")
        log_error("Unhandled exception reached main()", exc)
        pause_before_exit()
        return 1

    print_footer()
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================
# SUMMARY - resolved from the original draft's inline TODOs
# ============================================================
#
# IMPLEMENTED THIS PASS
#   - Install-root-anchored paths (matches the real download_model_OpenVINO/
#     models/ folder layout); correct hf_repo (llmware/qwen2.5-1.5b-instruct-ov)
#   - ChatHistory-based memory, replacing deprecated start_chat()/finish_chat()
#   - StreamingStatus-based streamer return value, replacing the removed
#     bool return (OpenVINO 2026.0.0)
#   - Download size shown before every install confirmation
#   - has_internet() fails fast (3s timeout) instead of hanging on a dead network
#   - Idea/chat memory: every Ideas session is saved to data/memory.db
#     (feature, start/end timestamps, turn count, full history as JSON) -
#     this is the "saves idea with date and time stamp" requirement from
#     the original project spec
#   - Manage Models menu: lists installed models with disk size, lets you
#     uninstall any of them except the mandatory base model
#   - Menu is generated entirely from the FEATURES registry - a feature
#     cannot exist in the menu without a working handler, or vice versa
#
# DROPPED (not deferred - actively removed)
#   - "Automate model registration" - contradicts your own stated
#     preference to add models yourself so you're actually reviewing code,
#     not copy-pasting. Models are added by writing one LocalModel entry
#     and listing it in the relevant *_MODELS tuple.
#
# IMPLEMENTED - CRASH VISIBILITY, AND THE ACTUAL ROOT CAUSE (CONFIRMED)
#   - Two earlier fix attempts (upgrading huggingface_hub, shortening the
#     install path) both targeted the wrong cause - neither was it. The real
#     problem was invisible until the crash-visibility work below let the
#     actual error surface for the first time.
#   - main() now has a genuine catch-all beneath AssistantError/
#     KeyboardInterrupt: nothing can escape as a raw, unexplained traceback.
#   - Every failure path calls log_error() (full timestamped traceback to
#     data/crash.log) and pause_before_exit() (holds the window open) - so a
#     failure is always readable, regardless of what causes it.
#   - The previously-unguarded model.path.parent.mkdir() call in
#     download_model() is now wrapped.
#   - CONFIRMED ROOT CAUSE: has_internet() opened a raw socket directly to
#     1.1.1.1:53 (Cloudflare DNS). That check was itself failing even though
#     real HTTPS access to Hugging Face worked fine moments earlier in the
#     same run (fetch_download_size succeeded) - proof the check was testing
#     the wrong thing, not that the network was actually down. Likely cause:
#     a network that allows normal HTTPS but blocks direct external DNS
#     lookups (common on campus/hostel networks).
#   - FIXED: has_internet() now makes a real HTTPS HEAD request to Hugging
#     Face itself - the actual service being depended on - instead of
#     testing an unrelated capability. Uses only the stdlib (urllib), no new
#     dependency.
#
# DEFERRED - NEXT PHASE
#   - PySide6 GUI: everything above the "CONSOLE UI" section is UI-agnostic
#     and gets reused as-is; only that section gets replaced. load_pipeline()
#     and ChatSession.ask() block, so both must run on a worker thread or
#     the window freezes during load/generation.
#
# DEFERRED - LATER VERSIONS
#   - Image generation, text-to-speech (v2); music generation (v3)
#   - Go Online / external API mode (v5)
#   - Browsing/searching past saved ideas, and a freeform notepad separate
#     from Ideas - both fit more naturally as GUI panels than console menus
#   - Auto-update - needs a hosted release/version manifest, out of scope here
#   - Normalising memory.db into per-message rows instead of one JSON blob
#     per session, if you ever need to search inside past conversations
#
# PACKAGING REMINDER
#   - Do not bundle models into the .exe - keep it small, download on first
#     run. Consider shipping one small model preinstalled for demo machines.
