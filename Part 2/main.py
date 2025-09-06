import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import os

# Initial Setup: NLTK Stopwords
# A one-time check and download for the NLTK stopwords list.
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Processes a raw tweet string to prepare it for analysis.

    The function must perform the following sequential operations:
    1. Remove any URLs or mentions (@username).
       - use the regular expression: r'http\S+|www\S+|@\w+'.

    2. Remove hashtag symbols (#) but keep the word.
    3. Convert the text to lowercase.
    4. Remove punctuation, digits, or non-alphabetic characters.
    5. Split into tokens (words).
    6. Remove stopwords.
    7. Rejoin the cleaned tokens into a single string.

    Args:
        text (str): The raw tweet text.

    Returns:
        str: The fully processed tweet string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs and mentions
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)

    # 2. Remove hashtag symbols (#)
    text = text.replace('#', '')

    # 3. Convert to lowercase
    text = text.lower()

    # 4. Remove punctuation, digits, or non-alphabetic characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 5. Tokenize
    tokens = text.split()

    # 6. Remove stopwords
    tokens = [word for word in tokens if word not in STOP_WORDS]

    # 7. Rejoin
    return " ".join(tokens)


if __name__ == "__main__":
    """
    Main execution function to run the hate dataset cleaning pipeline.
    """
    INPUT_FILE = os.path.join(os.path.dirname(__file__), 'hate_dataset.csv')
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'cleaned_hate_dataset.csv')

    # 1. Load the dataset from INPUT_FILE.
    # Handle the case where the file might not be found.
    print(f"Loading data from '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{INPUT_FILE}'.")
        print("Please ensure the dataset is in the correct directory.")
        exit(1)

    # 2. Apply the cleaning function to the 'tweet' column.
    print("Cleaning tweets...")
    df['cleaned_tweet'] = df['tweet'].apply(clean_text)

    # 3. Balance the dataset:
    #    - Keep all class 0 (Hate Speech) and class 2 (Neutral).
    #    - Sample 3500 rows from class 1 (Offensive).
    print("Balancing dataset...")
    df_hate = df[df['class'] == 0]
    df_offensive = df[df['class'] == 1]
    df_neutral = df[df['class'] == 2]

    df_offensive_sampled = df_offensive.sample(n=3500, random_state=42)

    # After balancing
    balanced_df = pd.concat([df_hate, df_offensive_sampled, df_neutral], axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. Save only 'cleaned_tweet' and 'class' columns.
    print(f"Saving cleaned and balanced data to '{OUTPUT_FILE}'...")
    balanced_df[['cleaned_tweet', 'class']].to_csv(OUTPUT_FILE, index=False)

    print("Data cleaning and balancing complete.")
    print(f"Final dataset saved to {OUTPUT_FILE}")

    print(balanced_df.tail(5))