def build_prompt_zero_shot(idea: str, note: str) -> str:
    """
    Balanced prompt: allow paraphrases & semantic matches,
    but forbid vague partial hints or inferred meaning.
    """
    idea = idea.strip()
    note = note.strip()

    return f"""You are determining whether a student's note contains a specific idea from a lecture.

Notes may contain shorthand, spelling errors, or informal phrasing. You SHOULD allow:
• Paraphrasing (different wording but same meaning)
• Misspellings that clearly refer to the same term
• Abbreviations that are unambiguous
• Conceptually equivalent statements

However, do NOT infer meaning that is not actually there. The idea is NOT present when:
• Only one or two keywords appear without expressing the idea
• The note vaguely hints at the topic but not the relationship or meaning
• The note mentions related concepts but not the actual idea
• Meaning is incomplete or ambiguous

Decision rules:
- Answer YES if the note expresses the same idea in any clear form.
- Answer NO if the idea is missing, incomplete, vague, or only suggested indirectly.

IMPORTANT: When you are not completely certain that the idea is expressed,
you MUST answer "NO". Only answer "YES" when the note clearly includes the full idea.
Do not guess or infer missing information. When unsure, answer "NO".

Your answer must be exactly one token: YES or NO.

[QUERY]
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
    Balanced one-shot prompt with support for paraphrases
    but strictness against vague matches.
    """
    idea = idea.strip()
    note = note.strip()
    ex_idea = ex_idea.strip()
    ex_note = ex_note.strip()

    ex_label_str = "YES" if ex_label == 1 else "NO"

    return f"""You are determining whether a student's note contains a specific lecture idea.

Notes may include shorthand, misspellings, or paraphrases. You SHOULD allow:
• Different wording that conveys the same meaning
• Misspellings or informal abbreviations that clearly match the idea
• Equivalent statements or conceptually accurate summaries

However, the idea is NOT present when:
• Only keywords appear without conveying the idea's meaning
• The note only hints at the topic without the core concept
• The relationship or main point of the idea is missing
• The wording is too incomplete to express the idea

IMPORTANT: When you are not completely certain that the idea is expressed,
you MUST answer "NO". Only answer "YES" when the note clearly includes the full idea.
Do not guess or infer missing information. When unsure, answer "NO".

Here is one labeled example from the same topic:

[EXAMPLE]
IDEA:
{ex_idea}

NOTE:
{ex_note}

Correct label: {ex_label_str}

Now apply the same rules to the new note.

[QUERY]
IDEA:
{idea}

NOTE:
{note}

Your final answer must be exactly one token: YES or NO.

Answer:"""
