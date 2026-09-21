import sys
import os
import time
import subprocess
from pathlib import Path  # Claude - comment: needed for install-root-anchored paths, replacing bare relative strings

try:
    import openvino
    import openvino_genai as ov_genai
    from openvino import Core

    core = Core()

except ImportError:
    print("OpenVINO isn't installed!")
    choice = input("Install it now automatically via pip? (y/n): ").strip().lower()

    if choice == "y":
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "openvino", "openvino-genai"]
            )
            # re-import after install so the rest of the script can use them
            import openvino
            import openvino_genai as ov_genai
            from openvino import Core

            core = Core()
            print("\nOpenVINO installed successfully. Continuing...\n")

        except subprocess.CalledProcessError as e:
            print(f"\nAuto-install failed: {e}")
            print("Please run manually: pip install openvino openvino-genai")
            sys.exit(1)
    else:
        print("To install Run CMD:")
        print("pip install openvino openvino-genai")
        sys.exit(1)


# -- Import ends and codebase starts! --
# -- Tried and worked for 2 whole hrs for refactoring this.. --

# ============================================================================
# Claude - comment: INSTALL ROOT RESOLUTION
# password_gate.cpp now launches this script with its own resolved exe
# directory as argv[1] (see the C++ fix from earlier), so both halves of the
# app agree on one "where do I live" answer instead of guessing separately.
# Falls back to this script's own folder when run standalone during dev
# (e.g. testing this file directly without going through the C++ gate),
# so nothing breaks while you're iterating on this half alone.
# ============================================================================
if len(sys.argv) > 1:
    INSTALL_ROOT = Path(sys.argv[1]).resolve()
else:
    INSTALL_ROOT = Path(__file__).resolve().parent

MODELS_DIR = INSTALL_ROOT / "download_model_OpenVINO" / "models"  # Claude - comment: FIXED - was INSTALL_ROOT/"models", one level too shallow. This now matches your own FOR THIS FOLDER SPECIFICALLY.txt, which specifies models live nested under download_model_OpenVINO/models/, not directly under the install root.


def banner():
    print("=" * 40)
    print("|" + " " * 38 + "|")
    print("|" + " " * 6 + "Welcome - Lemun Enterprises" + " " * 5 + "|")
    print("=" * 40)
    return

def devices():
    try:
        print("OpenVINO version:", openvino.__version__)
        print("Available devices:", core.available_devices)
    except Exception as e:
        print(f"Couldn't read device info: {e}")
    return


# -- NOTE that this "localmodels" function must be updated - if suppose you add 16Billion parameter model

def localmodels():
    # Claude - comment: paths are now built from MODELS_DIR (install-root-anchored,
    # nested under download_model_OpenVINO/models/ per your own folder convention)
    # instead of a hardcoded relative string, and each entry carries its real
    # Hugging Face repo id under "hf_repo" so ensure_base_model() below knows
    # exactly what to fetch — matches your own verified-working download script
    # at download_model_OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov.py exactly.

    # =================================
    # after choosing a model - if model not availble ask user confimation then install - me comment
    # =================================
    MODELS = {
                # == DEFAULT MODEL BELOW ==
                # == MUST BE DOWNLOADED DURING 1st INSTALLATION OF THE MODEL - HENCE INTERNET FOR THE FIRST TIME DURING APP DOWNLOAD IS MANDATORY
                "Qwen2.5-1.5B-Instruct-int4-ov":
                {"name":"Qwen2.5-1.5B-Instruct-int4-ov",
                 "company":"Alibaba",
                 "country":"China",
                 "path": str(MODELS_DIR / "Qwen2.5-1.5B-Instruct-int4-ov"),  # Claude - comment: was a bare relative string, now install-root-anchored
                 "hf_repo": "llmware/qwen2.5-1.5b-instruct-ov",              # Claude - comment: FIXED - was a different repo I'd found while researching (OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov), which doesn't match what your own download_model_OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov.py actually uses. This now matches your verified-working script exactly, so the automatic first-run download and your manual per-model scripts always agree.
                 "supports":"CPU,GPU"
                 }
            #   # IF ADDED YOU ADD MODEL HERE MANUALLY!! yeah pls no shit do some hardwork man... - its your flagship project - im lazy will update in v2 with automation.
            #   ,"qwen2.5-coder-3b":
            #   {"name": "Qwen2.5 Coder 3B",
            #    "company":"Alibaba",
            #    "country":"China",
            #    "path": "download_model_OpenVINO/models/Qwen2.5-Coder-3B-Instruct-int4-ov",
            #    "supports": "CPU,GPU"
            #    # yes all supports of models were tested beforehand on ASUS VIVOBOOK S14 S5406SA Intel Core Ultra 5 226v w/ 130V ARC IGPU + 40 TOPS NPU
            #    },
            #   "qwen3-8b":
            #   {"name": "Qwen3 8B",
            #    "company":"Alibaba",
            #    "country":"China",
            #    "path": "download_model_OpenVINO/models/Qwen3-8B-int4-cw-ov",
            #    "supports": "CPU,GPU,NPU"
            #    },
            #    "Llama-3.2-1B-Instruct-int4-ov":
            #    {"name": "Llama-3.2-1B-Instruct-int4-ov",
            #     "company":"Meta",
            #     "country":"United States",
            #     "path":"download_model_OpenVINO/models/Llama-3.2-1B-Instruct-int4-ov",
            #     "supports":"CPU,GPU"
            #     }
              }
    # returning it now instead of just printing so main() can actually use it
    return MODELS


# ============================================================================
# Claude - comment: MANDATORY FIRST-RUN BASE MODEL DOWNLOAD
# This is the piece that turns "download_model_OpenVINO/models/... must exist
# somehow" (the old TODO comment at the bottom of this file) into something
# that actually runs. It blocks here on purpose — no menu, no chat, until the
# default model is verifiably present — because the beta requirement is that
# first install cannot proceed without it.
# ============================================================================

def _model_is_present(model_path: Path) -> bool:
    # Claude - comment: OpenVINO IR models are a folder of files, not one
    # single file — checking for openvino_model.xml specifically (rather than
    # just the folder existing) catches a partial/interrupted previous
    # download instead of trusting an empty or half-written folder.
    return (model_path / "openvino_model.xml").exists()


def ensure_base_model(model_info):
    model_path = Path(model_info["path"])

    if _model_is_present(model_path):
        return  # already downloaded, nothing to do

    print(f"\nBase model '{model_info['name']}' not found - this is required before the app can run.")
    print("Downloading now (this only happens once)...\n")

    try:
        import huggingface_hub
    except ImportError:
        # Claude - comment: same auto-install pattern already used for OpenVINO
        # above, kept consistent rather than introducing a different style
        print("The 'huggingface_hub' package is required to download the base model.")
        choice = input("Install it now automatically via pip? (y/n): ").strip().lower()
        if choice != "y":
            print("Cannot continue without the base model. Exiting.")
            sys.exit(1)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            import huggingface_hub
        except subprocess.CalledProcessError as e:
            print(f"\nAuto-install failed: {e}")
            print("Please run manually: pip install huggingface_hub")
            sys.exit(1)

    model_path.parent.mkdir(parents=True, exist_ok=True)  # Claude - comment: ensure models/ exists before the download writes into it

    try:
        # Claude - comment: snapshot_download handles partial-download resume
        # and file integrity on its own - more reliable here than hand-rolling
        # this with raw requests, and it prints its own progress bars.
        huggingface_hub.snapshot_download(
            repo_id=model_info["hf_repo"],
            local_dir=str(model_path),
        )
    except Exception as e:
        print(f"\nBase model download failed: {e}")
        print("Check your internet connection and try again - the app cannot run without this model.")
        sys.exit(1)

    if not _model_is_present(model_path):
        print("\nDownload finished but the model files look incomplete. Please try again.")
        sys.exit(1)

    print(f"\nBase model '{model_info['name']}' downloaded successfully.\n")


# +++++++++++++++++++++++++++++++++++++++++++++++++
# me comment 
# CODEBASE TO UPDATE HERE PLS
# USER TO CHOOSE OPTIONS THEN REDIRECT TO MODELS
# OPTIONS INCLUDE
# 1. IDEAS - base model accepted - for text and knowledge share - since base model is always available no need to check
# 2. IMG GEN - user describes - go for quality - best roi
# 3. TEXT TO SPEECH 
# 4. MUSIC GEN
# 5. GO ONLINE! (Using API of course)
# 5. ADDITIONAL FEATURES - IF ADDED UPDATE HERE

# +++++++++++++++++++++++++++++++++++++++++++++++++
# - SAMPLE - 
# def menu():
#     print("Select what would you like to do\n1. Ideas\n2. Image Generation\n3. Music Generation\n4. Text To Speech\n5. Go Online! - Requires your API KEY!")
#     footer("Note that you can safely remove your API key as its stored in - data folder")
#     n=int(input("Choose a number between 1 to 5 : "))
#     # FOR NOW DO THIS - WE HAVENT IMPLEMENTED IT IN QT YET
#     # if(n==1):
#     #     return ideas
#     # if(n==2):
#     #     return Image_Gen
#     # if(n==3):
#     #     return Music_Gen
#     # if(n==4):
#     #     return Text_Speech
#     # if(n==5):
#     #     return Go_Online
#     return


# WILL REFACTOR LATER
def modelchoice(models):
    # user chooses models per session instead of every prompt -
    # so basically they choose model only when they restart the application - no model change allowed until app closed. - for now follow this we will update later

    print("\nAvailable Models:")
    keys = list(models.keys())
    for i, key in enumerate(keys, start=1):
        info = models[key]
        print(f"  {i}. {info['name']} - {info['company']} - {info['country']} - (supports: {info['supports']})")

    model = None
    while model is None:
        try:
            raw = input("\nPick a model (number): ").strip()
            idx = int(raw)
            if 1 <= idx <= len(keys):
                model = models[keys[idx - 1]]
            else:
                print(f"Enter a number between 1 and {len(keys)}.")
        except ValueError:
            print("That's not a number, try again.")

    # user to pick between cpu,gpu,npu
    supported = [d.strip().upper() for d in model["supports"].split(",")]
    detected = core.available_devices

    device = None
    while device is None:
        raw = input(f"Pick a device ({'/'.join(supported)}): ").strip().upper()
        if raw not in supported:
            print(f"'{model['name']}' only supports: {', '.join(supported)}")
            continue
        if not any(raw in d for d in detected):
            print(f"Heads up: OpenVINO doesn't see a {raw} on this machine ({detected}).")
            confirm = input("Try loading it anyway? (y/n): ").strip().lower()
            if confirm != "y":
                continue
        device = raw

    return model, device


def load_pipeline(model, device):
    print(f"\nLoading {model['name']} on {device}...")
    load_start = time.perf_counter()

    try:
        pipe = ov_genai.LLMPipeline(model["path"], device)
    except Exception as e:
        print(f"\nFailed to load model: {e}")
        print("Check that the model path exists and the device string is valid.")
        # WHEN creating .exe be CAREFUL - THE MODEL MOSTLY WONT BE BUNDLED WITH THE EXE since it would make it very big - so we need a way to download the model too - after installation
        # FOR TESTING PURPOSES AND FOR DEMO - people who like to test softwares - I recommmend that you keep a simple model like basic model bundled in - so it comes with atleast 1 model preinstalled.
        # ALSO THE USER SHOULD HAVE THE OPTION TO DYNAMICALLY UNINSTALL THE MODELS HE doesnt want - unnecessary space hogging.
        sys.exit(1)

    print(f"Model loaded in {time.perf_counter() - load_start:.2f} seconds")
    return pipe


def getresponse(pipe, chat_history, prompt):
    # reveives input of user from userprompts function()
    # this gets response from the localllm

    # Claude - comment: start_chat()/finish_chat() is deprecated in current
    # OpenVINO GenAI - the pipeline no longer holds conversation memory
    # internally. We now hold it ourselves in chat_history and pass the
    # whole thing into generate() every turn, per Intel's current chat-scenario
    # docs (openvinotoolkit.github.io/openvino.genai/docs/guides/chat-scenario).
    chat_history.append({"role": "user", "content": prompt})

    # dynamic tokens: we don't force a fixed length, the model stops itself on EOS.
    # max_new_tokens here is just a safety ceiling so one bad turn can't run forever.
    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = 2048

    print("Assistant: ", end="", flush=True)
    start = time.perf_counter()

    def streamer(subword):
        print(subword, end="", flush=True)
        return False  # False = keep generating, True would stop early

    try:
        result = pipe.generate(chat_history, gen_config, streamer)
        # Claude - comment: generate() still streams token-by-token via the
        # callback above exactly as before - streamer works the same way
        # regardless of whether the input is a plain string or a ChatHistory.
        # The difference is we now also get the final text back afterward,
        # which we need to append as the assistant's turn.
        chat_history.append({"role": "assistant", "content": result.texts[0]})
    except Exception as e:
        print(f"\n[Generation error: {e}]")
    finally:
        elapsed = time.perf_counter() - start
        print(f"\n[{elapsed:.2f}s]\n")


def userprompts(pipe):
    # this gets the user prompt - passes it to the getresponse function()
    # user can select the cpu/gpu/npu -> handled earlier in modelchoice(), one time per session

    print("\nChat session started. Type 'exit' or 'quit' to end.\n")

    # Claude - comment: start_chat() is deprecated - we now own the
    # conversation ourselves via ChatHistory and pass it into every
    # generate() call instead. No matching finish_chat() call needed below
    # since there's no pipeline-side chat state left to close out.
    chat_history = ov_genai.ChatHistory()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted.")
            break

        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue

        getresponse(pipe, chat_history, prompt)


def footer():
    print("Session ended. Thanks for using the assistant.")
    print("Copyright Product : Avoid Unfair Use and Follow the Code of Conduct")
    # yeah write some crap here lol
    return


def main():
    banner()
    devices()

    models = localmodels()

    # Claude - comment: default/base model is always the first entry in the
    # dict (matches the existing "only one uncommented entry" convention) -
    # forced download happens here, before the model picker is ever shown,
    # per the beta requirement that first install cannot proceed without it.
    default_model = next(iter(models.values()))
    ensure_base_model(default_model)

    model, device = modelchoice(models)

    pipe = load_pipeline(model, device)

    userprompts(pipe)

    footer()
    return

# NOTE - RESPONSE IS GIVEN TOKEN BY TOKEN!! - just like other llm's

if __name__ == "__main__":
    main()




# user gives the prompt - the model should remember the chat history for the session only for now. -- DONE, now via a ChatHistory object passed into generate() each turn (Claude - comment: start_chat()/finish_chat() were deprecated by OpenVINO and have been removed, see getresponse()/userprompts())
# the model should use dynamic tokens for response as much as it sees fit. -- DONE, model self-stops on EOS, max_new_tokens is just a safety ceiling
# do something to make sure that the model updates accoringly by the internet. -- PENDINE - HOLD THIS UNTIL REFACTOR IS DONE AND ARCHITECTUEE CHANGE IS DONE
# fetches whatever it doeesnt know from the internet - always internet since always is connected that should be the priority - fallback patch the offline works as much as it knows to provide without the internet - dont implement api's yet - for now local models are enough
# ^ still not implemented on purpose per the note above - no APIs yet, local-only for this pass

# == FEATURE UPDATE - MEMORIES TO BE ADDED AS A DATABASE INSIDE THE DATA FOLDER WHERE THE KEY IS STORED - PENDING
# - DIRECTORY OF DATA FOLDER IS THE SAME FOLDER AS THIS .py file so .\data\key.txt -> if im not wrong - its in the same folder as .py file and the key is inside the data folder

# ================= FOLLOW THIS BELOW UPPER WHATEVER YOU READ WAS DRAFT COMMENT ==================
# -> for currently -> app runs (telling about this python file specifically as the .cpp file handles the authentication and the password) -> checks openvino -> try / catch forces openvino install -> openvino checks the stuffs npu,gpu,cpu
# -> after openvino device check -> check for default model -> if not there -> force download -> no internet -> prompt to connect to internet and try again (implement the same try / catch to prevent freeze - just check for wifi instead of tracking - thats eazy)
# -> after default model successfulyl installed -> open menu -> show available n options (may dyanmically change with future) -> user goes in one of the option -> model unavailble -> ask confirmation to install the model - show size as well so user can decide -> then fine!
# -> for now leave the go_online mode - yeah leave it for now -> we do it in v5

# -> for now skip these below -----
# -> implement memory for the llm - json format
# -> implement the notepad in the main menu
# -> implement auto update



# ===== IMPLEMENT LIKE THIS IF THIS IS READABLE - IDK IM NUB YOU TELL ME ======

def basemodel():
    # add base model here -> dictionary or whatveer - should always be installed first - if first time run - or if detected that the program doesnt have it
    return

# IGNORE THESE DUMMY VALUES - ILL TEST AND TAKE MODELS SPECICALLY

def ideamodel():
    # add idea models here - 8billon,32billion etc
    # MODELS = {"name":"Qwen2.5-1.5B-Instruct-int4-ov"}
    return

def imgmodel():
    # add image gen models here - any parameter
    # MODELS = {"name":"Qwen2.5-1.5B-Instruct-int4-ov"}
    return

# similary n functions may be there -> as the program grows and features grows



# dont worry about replacing the entire file - just keep the comments all located at the bottom so i can refer in the new file what all changed