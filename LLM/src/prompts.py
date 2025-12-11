def build_prompt_zero_shot(idea: str, note: str) -> str:
    """
    Zero-shot prompt with explicit guidance for noisy student notes.
    """
    idea = idea.strip()
    note = note.strip()

    return f"""You are grading whether a student's note contains a specific idea from a lecture.

The notes are often noisy: they may contain misspellings, abbreviations, partial words,
or informal shorthand (for example, "def" for "definition", "ex" for "example",
"ALU" for "arithmetic logic unit"). They can also omit function words, repeat phrases,
or mix relevant and irrelevant information.

Follow these rules:

1. Focus on meaning, not exact wording. If the note clearly expresses the same idea
   in different words, treat the idea as present.
2. If a word in the note is slightly misspelled but obviously refers to a term in the idea,
   treat it as a match.
3. If the note uses an abbreviation or acronym that unambiguously refers to the concept
   in the idea, treat it as a match.
4. The idea counts as present if the central concept and its main relationship are stated,
   even if some minor details are missing.
5. If the note only vaguely hints at the idea or mentions isolated words without conveying
   the main point, treat the idea as NOT present.
6. Do not assume missing information. Only consider what is explicitly stated in the note.

After applying these rules:
- If the idea is clearly present in the note, answer "YES".
- If the idea is missing or too incomplete, answer "NO".

Your final answer must be exactly one token: YES or NO.
Do NOT include any explanation or extra text.

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
    One-shot prompt with noise-handling rules and a labeled example.
    """
    idea = idea.strip()
    note = note.strip()
    ex_idea = ex_idea.strip()
    ex_note = ex_note.strip()

    ex_label_str = "YES" if ex_label == 1 else "NO"

    return f"""You are grading whether a student's note contains a specific idea from a lecture.

The notes are often noisy: they may contain misspellings, abbreviations, partial words,
or informal shorthand (for example, "def" for "definition", "ex" for "example",
"ALU" for "arithmetic logic unit"). They can also omit function words, repeat phrases,
or mix relevant and irrelevant information.

Follow these rules:

1. Focus on meaning, not exact wording. If the note clearly expresses the same idea
   in different words, treat the idea as present.
2. If a word in the note is slightly misspelled but obviously refers to a term in the idea,
   treat it as a match.
3. If the note uses an abbreviation or acronym that unambiguously refers to the concept
   in the idea, treat it as a match.
4. The idea counts as present if the central concept and its main relationship are stated,
   even if some minor details are missing.
5. If the note only vaguely hints at the idea or mentions isolated words without conveying
   the main point, treat the idea as NOT present.
6. Do not assume missing information. Only consider what is explicitly stated in the note.

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

Your final answer must be exactly one token: YES or NO.
Do NOT include any explanation or extra text.

Answer:"""
