import pandas as pd

# Mapping segment numbers to the note column names in Notes.csv
SEGMENT_COLS = {
    1: "Segment1_Notes",
    2: "Segment2_Notes",
    3: "Segment3_Notes",
    4: "Segment4_Notes",
}

def load_data(data_dir="data"):
    """
    Load Notes.csv, train.csv, and test.csv from the given directory.
    Example directory: LLM/data/
    """
    notes = pd.read_csv(f"{data_dir}/Notes.csv")
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")

    print("Data loaded successfully!")
    return notes, train, test


def add_segment_text(df, notes_df):
    """
    Add a new column named 'segment_text' to the train or test DataFrame
    by getting the correct note segment from Notes.csv based on
    Topic, ID, and Segment columns.
    """

    # Merge based on matching metadata
    merged = df.merge(
        notes_df,
        on=["Topic", "ID"],
        how="left",
        suffixes=("", "_note")  # avoid duplicate column names
    )

    # Get the correct segment column
    def _get_segment(row):
        seg = int(row["Segment"])
        col = SEGMENT_COLS.get(seg)
        # Return segment_text only if segment number is valid,
        # the segment column exists, and the value is not null
        if col and col in row and pd.notnull(row[col]):
            return row[col]
        return ""

    merged["segment_text"] = merged.apply(_get_segment, axis=1)
    print("New column 'segment_text' correctly added!")
    return merged
