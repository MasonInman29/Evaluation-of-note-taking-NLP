import pandas as pd

# Mapping segment numbers to the note column names in Notes.csv
SEGMENT_COLS = {
    1: "Segment1_Notes",
    2: "Segment2_Notes",
    3: "Segment3_Notes",
    4: "Segment4_Notes",
}


def load_data(data_dir: str = "data"):
    """
    Load Notes.csv, train.csv, and test.csv from the given directory.

    Uses a lenient encoding to avoid UnicodeDecodeError caused by
    smart quotes or other non-UTF-8 characters.
    """
    notes = pd.read_csv(f"{data_dir}/Notes.csv", encoding="latin1")
    train = pd.read_csv(f"{data_dir}/train.csv", encoding="latin1")
    test = pd.read_csv(f"{data_dir}/test.csv", encoding="latin1")

    print("Data loaded successfully!")
    return notes, train, test


def add_segment_text(df: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new column named 'segment_text' to the train or test DataFrame
    by getting the correct note segment from Notes.csv based on
    shared keys (Topic, ID, optionally Experiment) and Segment.
    """

    # Decide merge keys based on what both dataframes actually have
    merge_keys = ["Topic", "ID"]
    if "Experiment" in df.columns and "Experiment" in notes_df.columns:
        merge_keys.insert(0, "Experiment")

    merged = df.merge(
        notes_df,
        on=merge_keys,
        how="left",
        suffixes=("", "_note"),  # avoid duplicate column names
    )

    # Get the correct segment column
    def _get_segment(row):
        seg = int(row["Segment"])
        col = SEGMENT_COLS.get(seg)
        # Return segment_text only if segment number is valid,
        # the segment column exists, and the value is not null
        if col and col in row and pd.notnull(row[col]):
            return str(row[col])
        return ""

    merged["segment_text"] = merged.apply(_get_segment, axis=1)
    print("New column 'segment_text' correctly added!")
    return merged
