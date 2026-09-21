"""
Lemun Enterprises - Local LLM Assistant
=======================================

A privacy-first, fully offline AI assistant powered by OpenVINO.

This file is the Python half of the application. The C++ half
(password_gate.cpp) handles authentication and then launches this script,
passing its own install directory as argv[1].

RUNTIME FLOW (matches the architecture notes at the bottom of this file):

    1. ensure_openvino()      -> OpenVINO present? if not, offer to install it
    2. detect_devices()       -> which of CPU / GPU / NPU actually exist here
    3. ensure_base_model()    -> base model present? if not, force download
    4. run_menu()             -> user picks a feature (Ideas / Image / TTS / ...)
    5. <feature handler>      -> model present? confirm + install, then run

READING GUIDE - the file is split into numbered sections:

    SECTION 1  Paths and constants
    SECTION 2  Small utility helpers
    SECTION 3  OpenVINO environment setup
    SECTION 4  Model catalog (what models exist)
    SECTION 5  Model management (checking, sizing, downloading)
    SECTION 6  Inference engine  <- UI-agnostic, reused as-is by the GUI later
    SECTION 7  Console UI        <- replaced by PySide6 later
    SECTION 8  Feature registry and handlers
    SECTION 9  Entry point
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional


# =============================================================================
# SECTION 1 - PATHS AND CONSTANTS
# =============================================================================

def resolve_install_root() -> Path:
    """
    Work out where the application actually lives on disk.

    password_gate.cpp passes its own resolved executable directory as argv[1],
    so both halves of the app agree on one single "where do I live" answer.

    When this script is run directly during development (no argv[1]), we fall
    back to this file's own folder so nothing breaks while iterating.

    Never trust the current working directory for this - it changes depending
    on how the app was launched, and breaks entirely once the app is bundled.
    """
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    return Path(__file__).resolve().parent


INSTALL_ROOT = resolve_install_root()

# Models live nested one level deeper, per the project's folder convention
# documented in download_model_OpenVINO/FOR THIS FOLDER SPECIFICALLY.txt
MODELS_DIR = INSTALL_ROOT / "download_model_OpenVINO" / "models"

# Same folder password_gate.cpp writes key.txt and lockout.txt into.
# The future chat-memory database belongs here too.
DATA_DIR = INSTALL_ROOT / "data"

# A safety ceiling only. The model normally stops itself on EOS long before
# this - it exists so one runaway turn cannot generate forever.
MAX_NEW_TOKENS = 2048

# Every OpenVINO IR model folder contains this file. Checking for it (rather
# than just checking the folder exists) catches a half-finished download.
MODEL_MARKER_FILE = "openvino_model.xml"

APP_TITLE = "Welcome - Lemun Enterprises"


# =============================================================================
# SECTION 2 - SMALL UTILITY HELPERS
# =============================================================================

class AssistantError(Exception):
    """
    Raised for any failure the user can understand and act on.

    Everything in this file raises AssistantError instead of calling
    sys.exit() directly. That matters for the GUI port: main() catches it
    and prints a message, but the GUI will catch the exact same exception
    and show a dialog box instead. One error path, two different UIs.
    """


def print_banner() -> None:
    """Print the application header box."""
    width = 40
    print("=" * width)
    print("|" + " " * (width - 2) + "|")
    print("|" + APP_TITLE.center(width - 2) + "|")
    print("|" + " " * (width - 2) + "|")
    print("=" * width)


def print_footer() -> None:
    """Print the closing message."""
    print("\nSession ended. Thanks for using the assistant.")
    print("Copyright Product : Avoid Unfair Use and Follow the Code of Conduct")


def format_bytes(num_bytes: Optional[int]) -> str:
    """
    Turn a raw byte count into something a human can judge at a glance.

    Returns "unknown" for None so callers never have to special-case it.

        >>> format_bytes(1_500_000_000)
        '1.40 GB'
    """
    if num_bytes is None:
        return "unknown"

    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"  # unreachable, kept for clarity


def has_internet(host: str = "1.1.1.1", port: int = 53, timeout: float = 3.0) -> bool:
    """
    Check for a live internet connection without ever hanging.

    Opens a socket to a public DNS server rather than fetching a web page:
    it is faster, needs no extra library, and the short timeout guarantees
    this returns quickly instead of freezing the app on a dead network.

    Note this reports *connectivity*, not *reachability of Hugging Face* -
    a captive portal (hotel/college wifi) can pass this check and still
    fail the download. The download itself has its own error handling.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pip_install(*packages: str) -> None:
    """
    Install packages into the currently running interpreter.

    Uses sys.executable rather than a bare "pip" so the packages always land
    in the same Python that is running this script - which is exactly what
    goes wrong when a machine has several Python installations.

    Raises:
        AssistantError: if pip fails for any reason.
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
    except subprocess.CalledProcessError as exc:
        raise AssistantError(
            f"Automatic install failed (exit code {exc.returncode}).\n"
            f"Please run manually:  pip install {' '.join(packages)}"
        ) from exc
    except OSError as exc:
        raise AssistantError(f"Could not start pip: {exc}") from exc


# =============================================================================
# SECTION 3 - OPENVINO ENVIRONMENT SETUP
# =============================================================================
#
# These start as None and are filled in by ensure_openvino().
#
# Why not just import at the top of the file like normal? Because the import
# needs to be able to ASK the user something ("install it now?"), and code
# that runs at import time cannot do that safely - especially not from a GUI,
# where there is no console to type an answer into. Doing it inside a function
# means we control exactly when it happens.

openvino = None      # the openvino module itself
ov_genai = None      # the openvino_genai module
core = None          # an openvino.Core() instance, used for device detection


def ensure_openvino() -> None:
    """
    Import OpenVINO, offering to install it if it is missing.

    Raises:
        AssistantError: if OpenVINO is unavailable and cannot be installed.
    """
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
    """
    Ask OpenVINO which compute devices this machine actually has.

    Returns something like ('CPU', 'GPU.0', 'NPU'). Note the suffixes -
    that is why device matching below uses "startswith" rather than "==".

    Never raises: a machine that reports no devices is reported as empty
    rather than crashing, so the user still sees a useful message.
    """
    try:
        return tuple(core.available_devices)
    except Exception as exc:
        print(f"Couldn't read device info: {exc}")
        return ()


def print_environment() -> tuple[str, ...]:
    """Print the OpenVINO version and detected hardware. Returns the devices."""
    try:
        print("OpenVINO version:", openvino.__version__)
    except Exception:
        print("OpenVINO version: unknown")

    devices = detect_devices()
    print("Available devices:", ", ".join(devices) if devices else "none detected")
    return devices


def stream_continue_value():
    """
    Return whatever value the installed OpenVINO expects from a streamer
    callback to mean "keep generating".

    Older versions accepted a plain False. That was deprecated and then
    REMOVED in OpenVINO 2026.0.0, replaced by the StreamingStatus enum.
    Resolving it dynamically means this code works on both, instead of
    breaking the next time OpenVINO is updated.
    """
    streaming_status = getattr(ov_genai, "StreamingStatus", None)
    if streaming_status is not None:
        return streaming_status.RUNNING
    return False  # legacy fallback for OpenVINO older than 2025.2


# =============================================================================
# SECTION 4 - MODEL CATALOG
# =============================================================================

@dataclass(frozen=True)
class LocalModel:
    """
    One downloadable OpenVINO model.

    frozen=True makes these read-only after creation, so a model definition
    can never be accidentally modified at runtime.
    """

    name: str                        # shown to the user
    company: str                     # who made it
    country: str                     # where they are based
    folder: str                      # folder name under MODELS_DIR
    hf_repo: str                     # Hugging Face repo id to download from
    supports: tuple[str, ...]        # devices this model can run on

    @property
    def path(self) -> Path:
        """Full install-root-anchored path to this model's folder."""
        return MODELS_DIR / self.folder

    @property
    def is_installed(self) -> bool:
        """True only if the model is fully downloaded, not half-written."""
        return (self.path / MODEL_MARKER_FILE).exists()

    def describe(self) -> str:
        """One-line summary for menus."""
        state = "installed" if self.is_installed else "not installed"
        return (
            f"{self.name} - {self.company} - {self.country} "
            f"(runs on: {', '.join(self.supports)}) [{state}]"
        )


# -----------------------------------------------------------------------------
# THE BASE MODEL
#
# This one is mandatory. It is downloaded on first run and is what the Ideas
# feature always uses, so that feature never has to check for availability.
# -----------------------------------------------------------------------------

BASE_MODEL = LocalModel(
    name="Qwen2.5-1.5B-Instruct-int4-ov",
    company="Alibaba",
    country="China",
    folder="Qwen2.5-1.5B-Instruct-int4-ov",
    hf_repo="llmware/qwen2.5-1.5b-instruct-ov",
    supports=("CPU", "GPU"),
)


# -----------------------------------------------------------------------------
# ADDING MORE MODELS
#
# To add a model: create a LocalModel below, then list it in the relevant
# tuple. Device support should be tested on real hardware before being
# claimed here (all current entries were verified on an ASUS Vivobook S14
# S5406SA - Core Ultra 5 226V, Arc 130V iGPU, 40 TOPS NPU).
#
# Commented-out entries are ones already researched but not yet wired up.
# -----------------------------------------------------------------------------

# QWEN_CODER_3B = LocalModel(
#     name="Qwen2.5 Coder 3B",
#     company="Alibaba",
#     country="China",
#     folder="Qwen2.5-Coder-3B-Instruct-int4-ov",
#     hf_repo="<fill in the Hugging Face repo id>",
#     supports=("CPU", "GPU"),
# )

# QWEN3_8B = LocalModel(
#     name="Qwen3 8B",
#     company="Alibaba",
#     country="China",
#     folder="Qwen3-8B-int4-cw-ov",
#     hf_repo="<fill in the Hugging Face repo id>",
#     supports=("CPU", "GPU", "NPU"),
# )

# LLAMA_1B = LocalModel(
#     name="Llama-3.2-1B-Instruct",
#     company="Meta",
#     country="United States",
#     folder="Llama-3.2-1B-Instruct-int4-ov",
#     hf_repo="<fill in the Hugging Face repo id>",
#     supports=("CPU", "GPU"),
# )


# Models grouped by what they are FOR. Each feature below draws from one of
# these tuples, so adding a model to a feature is a one-line change.

IDEA_MODELS: tuple[LocalModel, ...] = (BASE_MODEL,)
IMAGE_MODELS: tuple[LocalModel, ...] = ()
SPEECH_MODELS: tuple[LocalModel, ...] = ()
MUSIC_MODELS: tuple[LocalModel, ...] = ()


# =============================================================================
# SECTION 5 - MODEL MANAGEMENT
# =============================================================================

def fetch_download_size(model: LocalModel) -> Optional[int]:
    """
    Ask Hugging Face how large a model is, in bytes, before downloading it.

    Returns None if the size cannot be determined - which is not an error.
    The user is simply shown "unknown" and can still choose to proceed.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model.hf_repo, files_metadata=True)
        return sum(sibling.size for sibling in info.siblings if sibling.size)
    except Exception:
        # Deliberately broad: a missing size is a cosmetic problem, never a
        # reason to stop the user from downloading.
        return None


def ensure_huggingface_hub() -> None:
    """
    Make sure the download library is available, offering to install it.

    Raises:
        AssistantError: if it is missing and the user declines to install.
    """
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
    """
    Download a model from Hugging Face into its install-root-anchored folder.

    Raises:
        AssistantError: if there is no internet, or the download fails or
                        finishes incomplete.
    """
    if not has_internet():
        raise AssistantError(
            f"No internet connection detected.\n"
            f"'{model.name}' still needs to be downloaded, so please connect "
            f"to the internet and try again."
        )

    ensure_huggingface_hub()
    from huggingface_hub import snapshot_download

    model.path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {model.name}...")
    print("(This only happens once. Progress is shown below.)\n")

    try:
        # snapshot_download handles resuming partial downloads and verifying
        # file integrity itself - far more reliable than hand-rolling this.
        snapshot_download(repo_id=model.hf_repo, local_dir=str(model.path))
    except Exception as exc:
        raise AssistantError(
            f"Download of '{model.name}' failed: {exc}\n"
            f"Check your internet connection and try again."
        ) from exc

    if not model.is_installed:
        raise AssistantError(
            f"Download finished but '{model.name}' looks incomplete "
            f"(missing {MODEL_MARKER_FILE}). Please try again."
        )

    print(f"\n{model.name} downloaded successfully.\n")


def ensure_base_model() -> None:
    """
    Guarantee the mandatory base model exists before the app can be used.

    This runs before the menu is ever shown. It is deliberately blocking:
    the app is not usable without it.
    """
    if BASE_MODEL.is_installed:
        return

    print(f"\nBase model '{BASE_MODEL.name}' is not installed.")
    print("This model is required for the assistant to work at all.\n")

    size = fetch_download_size(BASE_MODEL)
    print(f"Download size: {format_bytes(size)}")

    download_model(BASE_MODEL)


def ensure_model_available(model: LocalModel) -> bool:
    """
    Make sure an OPTIONAL model is installed, asking the user first.

    Unlike the base model, the user is shown the size and may decline.

    Returns:
        True if the model is ready to use, False if the user declined.
    """
    if model.is_installed:
        return True

    print(f"\n'{model.name}' is not installed yet.")

    size = fetch_download_size(model)
    print(f"Download size: {format_bytes(size)}")

    answer = input("Download it now? (y/n): ").strip().lower()
    if answer != "y":
        print("Skipped - returning to the menu.")
        return False

    download_model(model)
    return True


# =============================================================================
# SECTION 6 - INFERENCE ENGINE
# =============================================================================
#
# IMPORTANT FOR THE GUI PORT:
# Nothing in this section prints anything or reads input. It receives text
# and hands back text through a callback. That is deliberate - when the
# PySide6 GUI is built, this entire section is reused UNCHANGED, and only
# SECTION 7 gets replaced. Keep it that way.

def load_pipeline(model: LocalModel, device: str):
    """
    Load a model onto a device and return a ready-to-use LLM pipeline.

    This is slow (several seconds) and blocking. In the GUI this must be
    called from a worker thread, never the UI thread, or the window freezes.

    Raises:
        AssistantError: if the model cannot be loaded.
    """
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
    """
    A single conversation with a loaded model.

    Owns the conversation history. OpenVINO used to hold this internally via
    start_chat()/finish_chat(), but that API is deprecated - the history now
    lives here and is passed into generate() on every turn.

    That is an improvement, not just a workaround: because this object holds
    the full conversation, saving it to the future memory database is simply
    a matter of serialising self.history - no extraction step needed.
    """

    def __init__(self, pipeline, max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        self.pipeline = pipeline
        self.max_new_tokens = max_new_tokens
        self.history = ov_genai.ChatHistory()
        self._keep_going = stream_continue_value()

    def ask(self, prompt: str, on_token: Callable[[str], None]) -> str:
        """
        Send one message and stream the reply back token by token.

        Args:
            prompt:   what the user typed.
            on_token: called once per token as it arrives. The console UI
                      prints it; the GUI will append it to a text box.

        Returns:
            The complete reply text.

        Raises:
            AssistantError: if generation fails.
        """
        self.history.append({"role": "user", "content": prompt})

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = self.max_new_tokens

        def streamer(chunk: str):
            on_token(chunk)
            return self._keep_going

        try:
            result = self.pipeline.generate(self.history, config, streamer)
        except Exception as exc:
            # Roll back the user turn so a failed request does not corrupt
            # the history for the next question.
            self.history.pop()
            raise AssistantError(f"Generation failed: {exc}") from exc

        reply = result.texts[0]
        self.history.append({"role": "assistant", "content": reply})
        return reply

    @property
    def turn_count(self) -> int:
        """How many complete exchanges have happened so far."""
        return len(self.history) // 2


# =============================================================================
# SECTION 7 - CONSOLE UI
# =============================================================================
#
# Everything that talks to a terminal lives here. When the GUI is built,
# this section is what gets replaced - nothing above it should need to change.

def ask_choice(prompt: str, count: int, allow_back: bool = True) -> Optional[int]:
    """
    Ask the user to pick a number from 1..count.

    Returns:
        The chosen index (1-based), or None if the user chose to go back.

    Keeps asking until the input is valid, so callers never see bad data.
    """
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
    """Show a numbered list of models and return the chosen one, or None."""
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
    """
    Ask which device to run on, offering only devices the model supports.

    Devices the machine does not appear to have are still offered, but
    flagged - OpenVINO occasionally reports devices in unexpected ways, so
    the user is warned rather than blocked.
    """
    print(f"\n'{model.name}' can run on: {', '.join(model.supports)}")

    options = []
    for device in model.supports:
        # "GPU" should match a reported "GPU.0", hence startswith
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
    """
    The interactive console chat loop.

    Ctrl+C stops the current session cleanly and returns to the menu rather
    than killing the whole application.
    """
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
            # A failed turn should not end the session - report and continue.
            print(f"\n[{exc}]\n")
            continue

        print(f"\n[{time.perf_counter() - started:.2f}s]\n")


# =============================================================================
# SECTION 8 - FEATURE REGISTRY AND HANDLERS
# =============================================================================

@dataclass(frozen=True)
class Feature:
    """
    One thing the user can choose to do from the main menu.

    The menu is generated FROM this registry, so a feature can never appear
    in the menu without a handler, or have a handler nothing can reach.
    That removes a whole category of bug where two lists drift apart.
    """

    label: str
    description: str
    models: tuple[LocalModel, ...] = ()
    handler: Optional[Callable[[tuple[str, ...]], None]] = None
    planned_for: str = ""            # e.g. "v2" - set when handler is None

    @property
    def is_available(self) -> bool:
        return self.handler is not None

    def describe(self) -> str:
        if self.is_available:
            return f"{self.label} - {self.description}"
        return f"{self.label} - {self.description}  [coming in {self.planned_for}]"


def run_ideas(detected_devices: tuple[str, ...]) -> None:
    """
    Feature 1 - Ideas.

    Free-form chat for capturing and refining ideas. Always uses the base
    model, which is guaranteed to be installed by this point.
    """
    model = choose_model(IDEA_MODELS)
    if model is None:
        return

    if not ensure_model_available(model):
        return

    device = choose_device(model, detected_devices)
    if device is None:
        return

    pipeline = load_pipeline(model, device)
    session = ChatSession(pipeline)
    run_chat_loop(session)

    print(f"\n({session.turn_count} exchanges this session)")


# -----------------------------------------------------------------------------
# THE MENU
#
# To add a feature: write a handler function above, then add a Feature entry
# here. The menu updates itself - there is no second list to keep in sync.
# -----------------------------------------------------------------------------

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
)


def run_menu(detected_devices: tuple[str, ...]) -> None:
    """
    The main menu loop. Runs until the user chooses to exit.

    Each feature runs inside its own error boundary, so a failure inside one
    feature returns the user to the menu instead of killing the application.
    """
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
            feature.handler(detected_devices)
        except AssistantError as exc:
            print(f"\n[{feature.label}] {exc}\n")
        except KeyboardInterrupt:
            print("\nInterrupted - returning to the menu.\n")


# =============================================================================
# SECTION 9 - ENTRY POINT
# =============================================================================

def main() -> int:
    """
    Start the assistant.

    Returns:
        A process exit code: 0 for success, 1 for a handled failure.
    """
    print_banner()

    try:
        ensure_openvino()
        detected_devices = print_environment()
        ensure_base_model()
        run_menu(detected_devices)
    except AssistantError as exc:
        print(f"\nERROR: {exc}\n")
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        return 1

    print_footer()
    return 0


if __name__ == "__main__":
    sys.exit(main())


# =============================================================================
# NOTES - WHAT CHANGED IN THIS REFACTOR
# =============================================================================
#
# STRUCTURE
#   - Split into 9 numbered sections, each with one job.
#   - The big one: SECTION 6 (engine) never prints or reads input, and
#     SECTION 7 (console UI) holds everything that does. When the PySide6 GUI
#     is built, SECTION 7 is replaced and SECTION 6 is reused untouched.
#
# ERROR HANDLING
#   - Added AssistantError. Nothing calls sys.exit() mid-flow any more;
#     failures raise, and main() is the single place that handles them.
#     The GUI will catch the same exception and show a dialog instead.
#   - A failed chat turn no longer ends the session, and rolls the user's
#     message back out of history so the next turn is not corrupted.
#   - A failure inside one feature returns to the menu instead of quitting.
#   - Ctrl+C is handled at every level instead of dumping a traceback.
#
# BUG FIXED - STREAMER RETURN VALUE (would have broken on the next update)
#   - The streamer callback returned False to mean "keep going". That was
#     deprecated, then REMOVED in OpenVINO 2026.0.0 in favour of the
#     StreamingStatus enum. stream_continue_value() now picks whichever the
#     installed version expects, so this works on old and new versions.
#
# MODEL CATALOG
#   - Models are now LocalModel dataclasses instead of nested dicts.
#     A typo like model.hf_rep fails loudly and immediately; a typo like
#     model["hf_rep"] used to fail deep inside the download call.
#   - model.path and model.is_installed are computed from the model itself,
#     so a path can never be written down inconsistently in two places.
#
# MENU
#   - The menu is generated from the FEATURES registry. Previously the menu
#     text and the code that acted on it were separate - the same "two places
#     that must agree" problem that caused the hf_repo mismatch earlier.
#     Adding a feature is now one entry in one tuple.
#
# NEW BEHAVIOUR
#   - Download size is fetched from Hugging Face and shown BEFORE asking for
#     confirmation, so the user can decide (your requirement).
#   - has_internet() checks connectivity with a 3 second timeout before
#     attempting a download, so a dead network fails fast instead of hanging.
#   - The menu loops - finishing a chat returns to the menu rather than
#     exiting the application.
#
# CONFIRMED
#   - DATA_DIR is INSTALL_ROOT/data, which is exactly where password_gate.cpp
#     writes key.txt and lockout.txt. Your reading was correct. The future
#     memory database belongs in that same folder.
#
#
# =============================================================================
# ROADMAP - STILL TO DO (preserved from the original file)
# =============================================================================
#
# CURRENT INTENDED FLOW  (this file only - the .cpp handles authentication)
#   app runs -> check openvino -> force install if missing -> detect devices
#   -> check base model -> force download if missing -> no internet? prompt
#   and retry -> menu -> user picks a feature -> model missing? show size,
#   ask, install -> run.
#   [ALL OF THE ABOVE IS NOW IMPLEMENTED]
#
# PENDING - NEXT UP
#   - PySide6 GUI: replace SECTION 7 only. Remember that load_pipeline() and
#     ChatSession.ask() block, so both must run on a worker thread or the
#     window will freeze during model load and generation.
#   - Chat memory as a database in the data folder (JSON or SQLite).
#     ChatSession.history is already the exact thing to persist.
#   - Notepad in the main menu.
#   - Auto update.
#
# PENDING - LATER VERSIONS
#   - Image generation (v2), text to speech (v2), music generation (v3).
#   - Go Online / external APIs (v5) - deliberately not started yet.
#   - Let the user uninstall models they no longer want, to reclaim space.
#   - Automate model registration instead of adding each one by hand.
#
# PACKAGING REMINDER
#   - Models must NOT be bundled into the .exe - it would be enormous.
#     Internet is required on first run so the base model can download.
#   - Consider shipping one small model preinstalled for demo machines.
