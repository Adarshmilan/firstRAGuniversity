"""
STEP 5 - GENERATE THE ANSWER  (the "G" in RAG)
----------------------------
Goal: give Gemini the retrieved chunks + the question, and tell it to answer
ONLY from those chunks. This "grounding" stops it from inventing fees/rooms.

Needs GEMINI_API_KEY in the environment. It's stored in the .env file, so run with:
    uv run --env-file .env python -m rag.step5_generate "What is the B.Tech fee?"
"""
import re
import sys
import time
from datetime import datetime

from google import genai
from google.genai import errors, types

from rag.config import (
    FALLBACK_CONTACT, FALLBACK_MODELS, LLM_MODEL, MAX_RETRIES, MAX_TOKENS,
    RETRY_BASE_SECONDS,
)
from rag.step4_retrieve import retrieve

client = genai.Client()  # reads GEMINI_API_KEY from the environment

SYSTEM_PROMPT = f"""You are the Marwadi University Assistant, a helpful chatbot for students \
and parents of Marwadi University.

Rules:
- Answer ONLY using the information inside <context>. Do not use outside knowledge.
- If the answer is not in the context, say you don't have that information and \
suggest contacting {FALLBACK_CONTACT}.
- Be short and clear. Use bullet points for lists of times or fees.
- Always include exact numbers, times and room numbers from the context.
- Use the current date/time given to answer questions like "is the HOD free now?"."""


def build_context(hits):
    """Wrap each chunk in tags so the model knows where each piece starts and ends."""
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f'<doc id="{i}" source="{h["source"]}">\n{h["text"]}\n</doc>')
    return "\n\n".join(parts)


def to_gemini_history(history):
    """
    Our app stores chat as {"role": "user"/"assistant", "content": str}.
    Gemini wants Content objects with role "user"/"model" and a list of parts.
    """
    contents = []
    for msg in history or []:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    return contents


RETRYABLE_CODES = {429, 500, 503}  # rate limit / server busy


def _retry_delay(err, attempt):
    """Use the wait time Google suggests (e.g. 'retry in 12s') if present, else back off."""
    m = re.search(r"retry(?:Delay)?\W+(\d+(?:\.\d+)?)s", str(err), re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 1, 60)
    return RETRY_BASE_SECONDS * (2 ** attempt)


def generate_with_fallback(contents, config):
    """
    Call Gemini; on 429 (too many requests) / 5xx, wait and retry,
    then fall back to the next model in FALLBACK_MODELS.
    """
    models = [LLM_MODEL] + [m for m in FALLBACK_MODELS if m != LLM_MODEL]
    last_err = None
    for model in models:
        for attempt in range(MAX_RETRIES):
            try:
                return client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except errors.APIError as e:
                last_err = e
                if e.code == 404:  # model name not available for this key -> next model
                    break
                if e.code not in RETRYABLE_CODES:
                    raise
                # Daily quota used up: retrying the same model won't help today
                if "PerDay" in str(e) or "per day" in str(e).lower():
                    break
                if attempt < MAX_RETRIES - 1:
                    time.sleep(_retry_delay(e, attempt))
        print(f"[gemini] {model} unavailable ({last_err.code}), trying next model...",
              file=sys.stderr)
    raise last_err


def answer(question, history=None):
    """
    question: the user's latest message
    history:  optional list of earlier {"role": "user"/"assistant", "content": str}
              so follow-ups like "what about ECE?" work
    Returns (answer_text, hits)
    """
    hits = retrieve(question)
    now = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")

    user_message = (
        f"<context>\n{build_context(hits)}\n</context>\n\n"
        f"Current date/time: {now}\n\n"
        f"Question: {question}"
    )
    contents = to_gemini_history(history) + [
        types.Content(role="user", parts=[types.Part(text=user_message)])
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=MAX_TOKENS,
        temperature=0.2,  # low = stick to the facts, less "creative"
    )
    try:
        response = generate_with_fallback(contents, config)
    except errors.APIError as e:
        if e.code in RETRYABLE_CODES:
            msg = ("The AI service is busy right now (too many requests). "
                   f"Please try again in a minute, or contact {FALLBACK_CONTACT}.")
            return msg, hits
        raise

    # response.text is None if the answer was blocked by safety filters
    text = response.text or "Sorry, I couldn't generate an answer for that."
    return text, hits


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the B.Tech fee?"
    reply, hits = answer(q)
    print(reply)
    print("\nSources:", ", ".join(sorted({h["source"] for h in hits})))
