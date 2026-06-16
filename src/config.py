import os 
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

# API KEYS
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")

#MODELS

# Claude is split into two separate families for drift computation.
# CLAUDE_MODELS keeps all Claude models under the single "claude" provider
# key used in run logic and results CSVs.
CLAUDE_SONNET_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
]

CLAUDE_OPUS_MODELS = [
    "claude-opus-4-1-20250805",
]

CLAUDE_MODELS = CLAUDE_SONNET_MODELS + CLAUDE_OPUS_MODELS

# Chronological release order (oldest → newest) within each provider family.
OPENAI_MODELS = [
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
]

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

ALL_MODELS = {
    "claude": CLAUDE_MODELS,
    "openai": OPENAI_MODELS,
    "gemini": GEMINI_MODELS,
}

# Open-Weighted Models
LLAMA31_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-405B-Instruct",
]

LLAMA32_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

GEMMA_MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
]

HF_MODELS = {
    "llama31": LLAMA31_MODELS,
    "llama32": LLAMA32_MODELS,
    "gemma": GEMMA_MODELS
}

# Version ordering for longitudinal drift computation.
# Each key is a provider family; models are listed oldest-first.
# Used by visualization.compute_fairness_drift() — do not reorder without
# also updating the drift computation logic.
VERSION_ORDERING = {
    "openai": [
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
    ],
    "claude_sonnet": [
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-5-20250929",
    ],
    "claude_opus": [
        "claude-opus-4-1-20250805",
    ],
    "gemini": [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ],
    "llama31": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
    ],
    "llama32": [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ],
    "gemma": [
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
    ],
}

#REQUIREMENTS
PROMPTS_PER_MODEL = 1000
MAX_TOKENS = 500
OUTPUT_DIR = "outputs/results"
PLOTS_DIR = "outputs/plots"
TABLES_DIR = "outputs/tables"

for dir_name in [OUTPUT_DIR, PLOTS_DIR, TABLES_DIR]:
    os.makedirs(dir_name, exist_ok=True)

#DEMOGRAPHIC AXES 
DEMOGRAPHIC_AXES = [
    "age",
    "disability_status",
    "gender_identity",
    "nationality",
    "physical_appearance",
    "race_ethnicity",
    "race_x_gender",
    "race_x_ses",
    "religion",
    "ses",
    "sexual_orientation"
]
