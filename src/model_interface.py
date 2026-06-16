import requests
import time
import json
from src.config import (
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    HUGGINGFACE_API_KEY
)


_anthropic_client = None
_openai_client = None
_gemini_configured = False


def _get_anthropic_client():
    global _anthropic_client
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")
    if _anthropic_client is None:
        from anthropic import Anthropic

        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _configure_gemini():
    global _gemini_configured
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set")
    if not _gemini_configured:
        import google.generativeai as genai

        genai.configure(api_key=GOOGLE_API_KEY)
        _gemini_configured = True


#query claude
def query_claude(model_name, prompt, max_tokens=500, temperature=0.0):
    try:
        response = _get_anthropic_client().messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"Error: {e}"

#query chatgpt
def query_openai(model_name, prompt, max_tokens=500, temperature=0.0):
    try:
        response = _get_openai_client().chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=42,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def query_gemini(model_name, prompt, max_tokens=500, temperature=0.0):
    try:
        _configure_gemini()
        import google.generativeai as genai

        model = genai.GenerativeModel(model_name)

        # Safety filters kept at defaults so filter events are real signal.
        # Responses that are blocked are logged by the caller (main.py).
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"[BLOCKED: {response.prompt_feedback.block_reason}]"

        return response.text.strip() if response.text else "[No text returned]"
    except Exception as e:
        return f"Error: {e}"

# Hugging Face model querying
HF_API_BASE = "https://router.huggingface.co/v1/chat/completions"

def query_hf_model(model_name, prompt, max_tokens=500, use_chat_template=True, temperature=0.0):
    """
    Query Hugging Face models via the OpenAI-compatible chat completions endpoint.
    """
    try:
        if not HUGGINGFACE_API_KEY:
            return "Error: HUGGINGFACE_API_KEY not set"

        url = HF_API_BASE
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": 42,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=120)

        # Handle 503 "model loading"
        if response.status_code == 503:
            try:
                data = response.json()
                wait_time = data.get("estimated_time", 30)
            except Exception:
                wait_time = 30
            print(f"Model is loading, waiting {wait_time} seconds...")
            time.sleep(wait_time)
            response = requests.post(url, json=payload, headers=headers, timeout=120)

        # Handle non-200 responses
        if response.status_code != 200:
            try:
                data = response.json()
                err_obj = data.get("error", data)
                if isinstance(err_obj, dict):
                    msg = err_obj.get("message") or str(err_obj)
                else:
                    msg = str(err_obj)
                error_msg = msg or f"HTTP {response.status_code}"
            except Exception:
                error_msg = f"HTTP {response.status_code}: {response.text[:200] if response.text else 'Empty response'}"
            return f"Error: {error_msg}"

        if not response.text:
            return f"Error: Empty response from API (status {response.status_code})"

        try:
            result = response.json()
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON response - {str(e)}. Response text: {response.text[:200]}"

        if isinstance(result, dict) and "choices" in result and result["choices"]:
            choice = result["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()

        return str(result).strip()

    except requests.exceptions.Timeout:
        return "Error: Request timeout - model may be too slow or unavailable"
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed - {str(e)}"
    except Exception as e:
        return f"Error: {e}"

def query_model(provider, model_name, prompt, max_tokens=500, temperature=0.0):

    if provider == "claude":
        return query_claude(model_name, prompt, max_tokens, temperature=temperature)
    elif provider == "openai":
        return query_openai(model_name, prompt, max_tokens, temperature=temperature)
    elif provider == "gemini":
        return query_gemini(model_name, prompt, max_tokens, temperature=temperature)
    elif provider in ["llama31", "llama32", "gemma"]:
        return query_hf_model(model_name, prompt, max_tokens, use_chat_template=True, temperature=temperature)
    else:
        return f"Error: Unknown provider '{provider}'. Supported: claude, openai, gemini, llama31, llama32, gemma"
