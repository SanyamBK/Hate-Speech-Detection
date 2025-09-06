import pandas as pd
from typing import Dict
import os

def task_1_load_and_inspect(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from the specified CSV file.

    Args:
        file_path (str): Path to the CSV file (e.g., 'hate_dataset.csv').

    Returns:
        pd.DataFrame: The loaded data.
    
    Instructions:
        - Use pandas to read the CSV file.
        - If the file is not found, let it raise FileNotFoundError.
        - Return the DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def task_2_check_dataset_health(df: pd.DataFrame) -> Dict[str, int]:
    """
    Checks the dataset for size and missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Dict[str, int]: Dictionary with keys:
                        "total_tweets" (total rows),
                        "missing_tweets" (nulls in 'tweet'),
                        "missing_classes" (nulls in 'class').
    
    Instructions:
        - Compute total number of rows.
        - Count nulls in 'tweet' and 'class' columns.
        - Return these counts in a dict with exactly the above keys.
    """
    total_tweets = len(df)
    missing_tweets = df['tweet'].isnull().sum()
    missing_classes = df['class'].isnull().sum()

    return {
        "total_tweets": total_tweets,
        "missing_tweets": missing_tweets,
        "missing_classes": missing_classes,
    }

def task_3_analyze_class_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Counts the number of tweets of each class.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: Counts of each class label (0, 1, 2).
    
    Instructions:
        - Use pd.Series.value_counts() on the 'class' column.
        - Return the resulting Series.
    """
    return df['class'].value_counts()

def task_4_analyze_tweet_length(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates average word counts of tweets overall and by class.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Dict[str, float]: Dictionary with keys:
                          "overall_avg_word_count",
                          "avg_hate_word_count",
                          "avg_offensive_word_count",
                          "avg_neither_word_count".
    
    Instructions:
        - Add a column 'word_count' that counts words in each tweet.
        - Compute the mean word count overall.
        - Filter tweets by class (0, 1, 2) and compute each mean.
        - Return a dict with the exact keys above.
    """
    # Add word_count column
    df['word_count'] = df['tweet'].apply(lambda x: len(str(x).split()))

    # Compute averages
    overall_avg = df['word_count'].mean()
    avg_hate = df[df['class'] == 0]['word_count'].mean()
    avg_offensive = df[df['class'] == 1]['word_count'].mean()
    avg_neither = df[df['class'] == 2]['word_count'].mean()

    return {
        "overall_avg_word_count": overall_avg,
        "avg_hate_word_count": avg_hate,
        "avg_offensive_word_count": avg_offensive,
        "avg_neither_word_count": avg_neither,
    }

if __name__ == "__main__":
    try:
        FILE_PATH = os.path.join(os.path.dirname(__file__), 'hate_speech.csv')
        
        # Task 1
        df = task_1_load_and_inspect(FILE_PATH)
        print("Task 1: Data Loaded")
        print(df.head())
        print("\n")
        
        # Task 2
        print("Task 2: Checking Dataset Health")
        health = task_2_check_dataset_health(df)
        for key, val in health.items():
            print(f"{key.replace('_', ' ').title()}: {val}")
        print("\n")
        
        # Task 3
        print("Task 3: Class Distribution")
        dist = task_3_analyze_class_distribution(df)
        print("Tweet counts per class:")
        print(dist)
        is_balanced = "Yes" if abs(dist.get(0,0)-dist.get(1,0)) < 0.05*len(df) else "No"
        print(f"\nIs the dataset balanced (roughly)? {is_balanced}")
        print("\n")
        
        # Task 4
        print("Task 4: Tweet Length Analysis")
        length_stats = task_4_analyze_tweet_length(df)
        print(f"Overall Average Word Count: {length_stats['overall_avg_word_count']:.2f} words")
        print(f"Average Hate-speech Word Count: {length_stats['avg_hate_word_count']:.2f} words")
        print(f"Average Offensive Word Count: {length_stats['avg_offensive_word_count']:.2f} words")
        print(f"Average Neutral Word Count: {length_stats['avg_neither_word_count']:.2f} words")
        
    except FileNotFoundError:
        print(f"Error: The file '{FILE_PATH}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")