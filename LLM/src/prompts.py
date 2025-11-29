def build_prompt_zero_shot(idea: str, note: str) -> str:
    """
    Zero-shot prompt: only instruction + (IdeaUnit, note).
    """
    idea = idea.strip()
    note = note.strip()

    return f"""You are grading whether a student's note contains a specific idea from a lecture.

- If the idea is clearly present (even if the wording is different), answer "YES".
- If the idea is missing or only vaguely hinted at, answer "NO".

Reply with exactly one word: YES or NO.
Do NOT include any explanation or extra text.

IDEA:
{idea}

NOTE:
{note}

Answer:"""


def build_prompt_one_shot(
    idea: str,
    note: str,
    ex_idea: str,
    ex_note: str,
    ex_label: int,
) -> str:
    """
    One-shot prompt: includes a single labeled example from the same topic.
    ex_label is 1 or 0, converted to YES/NO.
    """
    idea = idea.strip()
    note = note.strip()
    ex_idea = ex_idea.strip()
    ex_note = ex_note.strip()

    ex_label_str = "YES" if ex_label == 1 else "NO"

    return f"""You are grading whether a student's note contains a specific idea from a lecture.

- If the idea is clearly present (even if the wording is different), answer "YES".
- If the idea is missing or only vaguely hinted at, answer "NO".

Here is one labeled example from the same topic:

[EXAMPLE]
IDEA:
{ex_idea}

NOTE:
{ex_note}

Correct label: {ex_label_str}

Now grade the new note.

[QUERY]
IDEA:
{idea}

NOTE:
{note}

Answer with exactly one word: YES or NO.
Do NOT include any explanation or extra text.

Answer:"""
