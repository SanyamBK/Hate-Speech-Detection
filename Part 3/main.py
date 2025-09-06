import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from nltk.corpus import stopwords
import nltk
from sklearn.utils import resample


# Initial Setup: NLTK Stopwords
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))

# File Paths
INPUT_FILE = 'cleaned_hate_dataset.csv'
OUTPUT_FILE = 'hate_speech_model.pkl'

def train_model():
    """Main function to train and save the hate speech classification model."""
    print("Starting Model Training")
    try:
        # Task 1: Load the Data
        df = pd.read_csv(INPUT_FILE)

        df_hate = df[df['class'] == 0]
        df_offensive = df[df['class'] == 1]
        df_neutral = df[df['class'] == 2]

        df_offensive_sampled = df_offensive.sample(n=len(df_hate), random_state=42)
        df_neutral_sampled = df_neutral.sample(n=len(df_hate), random_state=42)

        # After balancing
        balanced_df = pd.concat([df_hate, df_offensive_sampled, df_neutral_sampled], axis=0)

        # Ensure consistent ordering with expected output
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Task 2: Prepare Features and Target (continues from here)
        X = balanced_df['cleaned_tweet'].astype(str).apply(clean_text)
        y = balanced_df['class']

        # Task 3: Create Training and Testing Sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Task 4: Vectorize the Text
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Task 5: Build and Train the Neural Network
        model = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.01,
            max_iter=40,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.2,
            random_state=42,
            verbose=True
        )

        # Fit the model
        model.fit(X_train_vec, y_train)

        # Task 6: Evaluate Performance
        predictions = model.predict(X_test_vec)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        accuracy = accuracy_score(y_test, predictions)
        print(f"Test Accuracy: {accuracy:.4f}")

        # Task 7: Save the Artifacts
        artifacts = {'vectorizer': vectorizer, 'model': model}
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(artifacts, f)
            
        print(f"--- Model and Vectorizer saved successfully to {OUTPUT_FILE}! ---")
    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_FILE}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Prediction Example (This part is complete for you)

def clean_text(text: str) -> str:
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

def run_prediction_example():
    """Loads the saved model and runs predictions on sample tweets."""
    print("\n--- Running Prediction Example ---")
    try:
        with open(OUTPUT_FILE, 'rb') as f:
            artifacts = pickle.load(f)
        
        vectorizer = artifacts['vectorizer']
        model = artifacts['model']
        
        sample_tweets = [
            "I hate you so much! You're the worst person ever.",
            "Check out my new blog post at http://example.com",
            "Had a great time at the concert last night!"
        ]
        
        for tweet in sample_tweets:
            cleaned_tweet = clean_text(tweet)
            tweet_vec = vectorizer.transform([cleaned_tweet])
            prediction = model.predict(tweet_vec)[0]  # [0] to get single label
            
            print(f"\nTweet: '{tweet}'")
            print(f"Predicted Class: {prediction}")
            
    except FileNotFoundError:
        print(f"Could not load '{OUTPUT_FILE}'. Please train the model first by running the script.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    train_model()
    run_prediction_example()