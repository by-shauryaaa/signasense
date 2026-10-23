"""
SignaSense — Grammar Smoother
===============================
Converts raw sign language glosses (ASL word order) into fluent spoken English.

Two modes:
    1. Rule-based lookup (offline, instant, limited to known phrases)
    2. LLM API call (flexible, handles unseen combinations, requires internet)

Usage:
    from src.grammar_smoother import smooth

    smooth(["hello", "name", "what"])
    # → "Hello, what is your name?"
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Rule-Based Phrase Map ─────────────────────────────────────────────────────
# Maps tuples of glosses to natural English sentences.
# Matching is case-insensitive. Order matters.
PHRASE_MAP = {
    # Greetings & Politeness
    ("hello",): "Hello!",
    ("goodbye",): "Goodbye! Have a nice day.",
    ("thank_you",): "Thank you!",
    ("please",): "Please.",
    ("sorry",): "I am sorry.",
    ("welcome",): "You're welcome!",
    ("how_are_you",): "How are you doing?",
    ("nice_to_meet_you",): "Nice to meet you!",
    ("hello", "how_are_you"): "Hello, how are you?",
    ("hello", "nice_to_meet_you"): "Hello, nice to meet you!",
    ("hello", "name", "what"): "Hello, what is your name?",
    ("name", "what"): "What is your name?",
    ("you", "name", "what"): "What is your name?",
    ("me", "name"): "My name is...",

    # Responses & Descriptors
    ("yes",): "Yes.",
    ("no",): "No.",
    ("good",): "That's good.",
    ("bad",): "That's bad.",
    ("fine",): "I'm doing fine.",
    ("me", "fine"): "I am doing fine.",
    ("me", "fine", "thank_you"): "I am fine, thank you!",
    ("thank_you", "good"): "Thank you, that's great!",

    # Basic Needs & Requests
    ("help",): "I need help.",
    ("help", "please"): "Please help me.",
    ("help", "me", "please"): "Please help me.",
    ("water",): "I would like some water.",
    ("water", "please"): "Can I have some water, please?",
    ("drink",): "I want something to drink.",
    ("drink", "please"): "Can I have a drink, please?",
    ("eat",): "I want to eat.",
    ("eat", "please"): "I would like something to eat, please.",
    ("more",): "More, please.",
    ("more", "please"): "Can I have some more, please?",
    ("more", "water", "please"): "More water, please.",
    ("want", "more"): "I want more, please.",
    ("me", "want", "water"): "I would like some water.",
    ("me", "want", "eat"): "I want to eat.",
    ("me", "want", "drink"): "I want something to drink.",
    ("me", "want", "sleep"): "I want to go to sleep.",
    ("sleep",): "I need to sleep.",
    ("sleep", "please"): "I would like to sleep.",
    ("stop",): "Stop.",
    ("stop", "please"): "Please stop.",

    # Questions & Locations
    ("where", "bathroom"): "Where is the bathroom?",
    ("bathroom", "where"): "Where is the bathroom, please?",
    ("bathroom", "please"): "May I use the bathroom, please?",
    ("who", "you"): "Who are you?",
    ("where", "home"): "Where is your home?",
    ("where", "work"): "Where do you work?",
    ("where", "doctor"): "Where is the doctor?",
    ("why", "sad"): "Why are you feeling sad?",
    ("how", "you"): "How are you?",

    # Health & Emotions
    ("happy",): "I am happy!",
    ("sad",): "I feel sad.",
    ("tired",): "I am tired.",
    ("me", "tired"): "I am very tired.",
    ("me", "tired", "sleep", "want"): "I'm tired and want to sleep.",
    ("pain",): "I am in pain.",
    ("me", "pain"): "It hurts.",
    ("pain", "where"): "Where does it hurt?",
    ("doctor",): "I need a doctor.",
    ("doctor", "need"): "I need to see a doctor.",
    ("help", "doctor", "please"): "Please help me find a doctor.",
    ("sorry", "help", "please"): "I'm sorry, could you please help me?",
    ("i_love_you",): "I love you.",

    # People, Places & Time
    ("friend",): "You are my friend.",
    ("family",): "This is my family.",
    ("home",): "I am at home.",
    ("home", "want"): "I want to go home.",
    ("work",): "I am at work.",
    ("today",): "Today.",
    ("now",): "Right now.",
    ("today", "work"): "I have work today.",
    ("tomorrow", "home"): "I will be home tomorrow.",
    ("today", "good"): "Today is a good day.",
    ("today", "we", "work"): "We have work today.",
    ("we", "work"): "We are working.",
    ("arm", "today"): "Today.",
    ("hello", "today"): "Hello today!",
    ("thank_you", "friend"): "Thank you, my friend!",
    ("see", "later"): "See you later!",
    ("see", "tomorrow"): "See you tomorrow!",
    ("we", "go", "home"): "We are going home.",
    ("time", "go"): "It is time to go.",
    ("call", "me"): "Please call me.",
    ("finish", "work"): "I finished my work.",
}


def _normalize_glosses(glosses: list[str]) -> tuple[str, ...]:
    """Lowercase and strip whitespace from glosses."""
    return tuple(g.strip().lower() for g in glosses if g.strip())


def smooth_rule_based(glosses: list[str]) -> str:
    """
    Try to match the glosses against the phrase map.

    Falls back to joining glosses with spaces if no match is found.

    Parameters
    ----------
    glosses : list[str]
        Sequence of detected sign glosses.

    Returns
    -------
    text : str
        Fluent English sentence or raw gloss fallback.
    """
    normalized = _normalize_glosses(glosses)

    if not normalized:
        return ""

    # Exact match
    if normalized in PHRASE_MAP:
        return PHRASE_MAP[normalized]

    # Try matching subsequences (longest match first)
    best_match = None
    best_length = 0

    for phrase_key, sentence in PHRASE_MAP.items():
        key_len = len(phrase_key)
        if key_len > best_length:
            # Check if phrase_key is a subsequence of normalized
            for start_idx in range(len(normalized) - key_len + 1):
                if normalized[start_idx : start_idx + key_len] == phrase_key:
                    best_match = sentence
                    best_length = key_len

    if best_match:
        return best_match

    # Fallback: capitalize and join
    return " ".join(g.replace("_", " ") for g in normalized).capitalize() + "."


def smooth_llm(glosses: list[str], api_key: str = None) -> str:
    """
    Use an LLM API to convert glosses into fluent English.

    Requires an API key set via OPENAI_API_KEY environment variable
    or passed directly.

    Parameters
    ----------
    glosses : list[str]
        Sequence of detected sign glosses.
    api_key : str, optional
        OpenAI API key. Falls back to env variable.

    Returns
    -------
    text : str
        Fluent English sentence from the LLM.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("[WARN] No API key found. Falling back to rule-based smoothing.")
        return smooth_rule_based(glosses)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)

        gloss_text = " ".join(g.upper() for g in glosses)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You convert American Sign Language (ASL) glosses into natural, "
                        "fluent English sentences. ASL has its own grammar and word order "
                        "(often Topic-Comment or Object-Subject-Verb). Your job is to "
                        "produce a single natural English sentence. Return ONLY the "
                        "sentence, nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Glosses: {gloss_text}",
                },
            ],
            max_tokens=100,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[WARN] LLM call failed: {e}. Falling back to rule-based.")
        return smooth_rule_based(glosses)


def smooth(glosses: list[str], use_llm: bool = False, api_key: str = None) -> str:
    """
    Main entry point: convert glosses to fluent English.

    Parameters
    ----------
    glosses : list[str]
        Sequence of detected sign glosses.
    use_llm : bool
        If True, attempt LLM-based smoothing first.
    api_key : str, optional
        API key for the LLM service.

    Returns
    -------
    text : str
        Fluent English sentence.
    """
    if not glosses:
        return ""

    if use_llm:
        return smooth_llm(glosses, api_key)
    else:
        return smooth_rule_based(glosses)


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ["hello"],
        ["hello", "name", "what"],
        ["help", "need"],
        ["water", "please"],
        ["thank_you"],
        ["stop", "please"],
        ["food", "want"],
        ["sorry", "help", "please"],
        # Unknown combination → fallback
        ["hello", "water", "good", "stop"],
    ]

    print("=" * 60)
    print("  SignaSense — Grammar Smoother Test")
    print("=" * 60)

    for glosses in test_cases:
        result = smooth(glosses)
        print(f"  {str(glosses):45s} → {result}")

    print("=" * 60)
