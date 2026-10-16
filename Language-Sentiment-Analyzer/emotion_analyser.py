"""
Language Sentiment Analyser - Emotion Detection
================================================

CS21002 group project (machine-learning component).

We treat emotion detection as a supervised multi-class classification problem
(Mitchell's Task / Performance / Experience framework):

    Task (T):        given a short piece of English text x, assign it one label y
                     from {sadness, joy, love, anger, fear, surprise}.
                     We learn a function f: X -> Y.
    Performance (P): accuracy and macro F1 score on a test set the model has
                     never seen.
    Experience (E):  the DAIR.AI "emotion" dataset of ~20,000 labelled tweets.

Two classifiers are compared, both trained on the same TF-IDF features:
    1. Multinomial logistic regression (uses softmax to give six probabilities)
    2. Multinomial Naive Bayes (uses Bayes' theorem, treats words as independent)

How to run:
    pip install -r requirements.txt
    python emotion_analyser.py

Everything this script produces (charts, results) is saved to the "outputs"
folder next to this file.
"""

import os

import matplotlib
matplotlib.use("Agg")  # draw charts straight to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# The dataset stores each emotion as a number 0-5. This list converts the
# numbers back into names: label 0 = "sadness", label 1 = "joy", and so on.
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Folder where charts and the results summary are saved.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Fixed random seed so every run gives the same results (reproducibility).
RANDOM_SEED = 42

# Candidate settings we try on the VALIDATION set. The best one is then used
# for the final model. The test set is never used for choosing settings.
#   C     -> logistic regression: how strongly weights are allowed to grow.
#            Smaller C = more regularisation (simpler model, less overfitting).
#   alpha -> Naive Bayes smoothing: gives words never seen with an emotion a
#            small non-zero probability, so one unseen word can't zero out a class.
LOGREG_C_VALUES = [0.1, 1.0, 10.0]
NB_ALPHA_VALUES = [0.01, 0.1, 1.0]

# Our own sentences, used to check how the models handle text that isn't
# from the dataset (the "above and beyond" part of the report).
OWN_SENTENCES = [
    "I finally passed my driving test, best day ever!",
    "I can't stop crying since my dog died",
    "I love spending time with my family at Christmas",
    "The bus was late again and I'm absolutely furious",
    "I'm really nervous about my exam tomorrow",
    "Wow, I did not expect to see you here!",
    "I'm not happy with how this turned out",
]


# ---------------------------------------------------------------------------
# Step 1: Load the data (the "Experience")
# ---------------------------------------------------------------------------

def load_data():
    """
    Download the DAIR.AI emotion dataset from Hugging Face.

    It already comes split into train (16,000), validation (2,000) and
    test (2,000) tweets. We keep these splits exactly as they are.

    Returns three (texts, labels) pairs, where labels are numbers 0-5.
    """
    from datasets import load_dataset  # imported here so the rest of the file runs without it

    dataset = load_dataset("dair-ai/emotion")

    def split(name):
        return list(dataset[name]["text"]), np.array(dataset[name]["label"])

    return split("train"), split("validation"), split("test")


def describe_data(train_labels):
    """
    Print how many training examples there are for each emotion.

    This shows the class imbalance mentioned in the report: far more joy and
    sadness than love or surprise.
    """
    print("\nTraining examples per emotion:")
    counts = np.bincount(train_labels, minlength=len(EMOTIONS))
    for emotion, count in zip(EMOTIONS, counts):
        print(f"  {emotion:<9} {count:>6}  ({count / len(train_labels):.1%})")

    # Save the same information as a bar chart for the report.
    plt.figure(figsize=(7, 4))
    plt.bar(EMOTIONS, counts, color="#4C72B0")
    plt.title("Training examples per emotion")
    plt.ylabel("Number of tweets")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Step 2: Turn text into numbers with TF-IDF
# ---------------------------------------------------------------------------

def build_features(train_texts, val_texts, test_texts):
    """
    Convert each tweet into a vector of numbers using TF-IDF.

    TF  (term frequency):          how often a word appears in this tweet.
    IDF (inverse document freq.):  how rare the word is across all tweets.
    A word scores highly if it appears in this tweet but is rare overall,
    e.g. "furious" scores high, "the" scores low.

    IMPORTANT: the vectoriser learns its vocabulary and IDF values from the
    TRAINING set only (fit_transform). The validation and test sets are only
    transformed with what was learned, so no information leaks from them.
    """
    vectoriser = TfidfVectorizer(
        lowercase=True,   # "Happy" and "happy" count as the same word
        min_df=2,         # ignore words that appear in fewer than 2 tweets (typos, rare noise)
    )
    X_train = vectoriser.fit_transform(train_texts)
    X_val = vectoriser.transform(val_texts)
    X_test = vectoriser.transform(test_texts)

    print(f"\nVocabulary size: {len(vectoriser.vocabulary_)} words")
    return vectoriser, X_train, X_val, X_test


# ---------------------------------------------------------------------------
# Step 3: Choose settings on the validation set, then train the models
# ---------------------------------------------------------------------------

def make_logistic_regression(C):
    """
    Multinomial logistic regression.

    It learns one weight per word for EACH emotion. For a tweet it adds up the
    weights of its words to get six scores, then softmax turns those scores
    into six probabilities that sum to 1. The highest probability wins.

    class_weight="balanced" makes mistakes on rare emotions (love, surprise)
    count for more during training, so the model doesn't just ignore them.
    """
    return LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def make_naive_bayes(alpha):
    """
    Multinomial Naive Bayes.

    Uses Bayes' theorem: P(emotion | words) is proportional to
    P(emotion) x P(word1 | emotion) x P(word2 | emotion) x ...
    It is "naive" because it assumes every word is independent of the others,
    so it can't tell that "not happy" means something different from "happy".
    """
    return MultinomialNB(alpha=alpha)


def tune(name, make_model, values, X_train, y_train, X_val, y_val):
    """
    Try each candidate setting, train on the training set, score on the
    validation set using macro F1, and return the best setting.
    """
    print(f"\nTuning {name} on the validation set (macro F1):")
    best_value, best_score = None, -1.0
    for value in values:
        model = make_model(value).fit(X_train, y_train)
        score = f1_score(y_val, model.predict(X_val), average="macro")
        print(f"  setting = {value:<5} -> macro F1 = {score:.3f}")
        if score > best_score:
            best_value, best_score = value, score
    print(f"  best setting: {best_value}")
    return best_value


# ---------------------------------------------------------------------------
# Step 4: Evaluate on the unseen test set (the "Performance")
# ---------------------------------------------------------------------------

def evaluate(name, model, X_test, y_test):
    """
    Measure how well a trained model does on the test set.

    Accuracy:  percentage of tweets labelled correctly.
    Macro F1:  F1 score worked out for each emotion separately, then averaged,
               so rare emotions count just as much as common ones.
    Also prints precision, recall and F1 for every emotion, and saves a
    confusion matrix showing which emotions get mixed up.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print(f"\n=== {name}: test set results ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    report = classification_report(y_test, predictions, target_names=EMOTIONS, digits=3)
    print(report)

    # Confusion matrix: rows = true emotion, columns = predicted emotion.
    # Normalised by row, so each cell shows the share of that true emotion.
    matrix = confusion_matrix(y_test, predictions, normalize="true")
    display = ConfusionMatrixDisplay(matrix, display_labels=EMOTIONS)
    display.plot(cmap="Blues", values_format=".2f", colorbar=False)
    plt.title(f"{name} - confusion matrix (test set)")
    plt.tight_layout()
    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()

    return {"accuracy": accuracy, "macro_f1": macro_f1, "report": report}


def plot_comparison(results):
    """Bar chart comparing both models' accuracy and macro F1 side by side."""
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["macro_f1"] for n in names]
    x = np.arange(len(names))

    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.2, accuracies, width=0.4, label="Accuracy", color="#4C72B0")
    plt.bar(x + 0.2, f1s, width=0.4, label="Macro F1", color="#DD8452")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model comparison on the test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()


def top_words(model, vectoriser, n=10):
    """
    Show which words logistic regression relies on most for each emotion.

    Because the model is just a table of weights (word x emotion), the words
    with the largest weight for an emotion are the strongest signs of it.
    This makes the model easy to explain in the report.
    """
    words = np.array(vectoriser.get_feature_names_out())
    print(f"\nTop {n} words per emotion (logistic regression weights):")
    lines = []
    for index, emotion in enumerate(EMOTIONS):
        best = np.argsort(model.coef_[index])[::-1][:n]
        line = f"  {emotion:<9} {', '.join(words[best])}"
        print(line)
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Step 5: Try the models on our own sentences
# ---------------------------------------------------------------------------

def predict_emotion(text, model, vectoriser):
    """
    Predict the emotion of any piece of text.

    The text goes through the SAME TF-IDF vectoriser as the training data,
    then the model returns a probability for each emotion.
    Returns (best emotion, its probability, all probabilities).
    """
    features = vectoriser.transform([text])
    probabilities = model.predict_proba(features)[0]
    best = int(np.argmax(probabilities))
    return EMOTIONS[best], probabilities[best], probabilities


def test_own_sentences(models, vectoriser):
    """Run every model on our own example sentences and print the results."""
    print("\nOur own sentences (text outside the dataset):")
    for sentence in OWN_SENTENCES:
        print(f'\n  "{sentence}"')
        for name, model in models.items():
            emotion, confidence, _ = predict_emotion(sentence, model, vectoriser)
            print(f"    {name:<20} -> {emotion} ({confidence:.0%})")


def interactive_demo(model, vectoriser):
    """Type sentences and see the predicted emotion. Press Enter on an empty line to quit."""
    print("\nInteractive demo - type a sentence (empty line to quit).")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        emotion, confidence, probabilities = predict_emotion(text, model, vectoriser)
        print(f"  Predicted: {emotion} ({confidence:.0%})")
        # Show the full probability spread, highest first.
        for i in np.argsort(probabilities)[::-1]:
            print(f"    {EMOTIONS[i]:<9} {probabilities[i]:.0%}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load the data and look at the class balance.
    print("Loading the DAIR.AI emotion dataset...")
    (train_texts, y_train), (val_texts, y_val), (test_texts, y_test) = load_data()
    print(f"Train: {len(train_texts)}  Validation: {len(val_texts)}  Test: {len(test_texts)}")
    describe_data(y_train)

    # 2. Turn the tweets into TF-IDF feature vectors.
    vectoriser, X_train, X_val, X_test = build_features(train_texts, val_texts, test_texts)

    # 3. Pick the best setting for each model using the validation set,
    #    then train the final models on the training set.
    best_c = tune("Logistic regression", make_logistic_regression,
                  LOGREG_C_VALUES, X_train, y_train, X_val, y_val)
    best_alpha = tune("Naive Bayes", make_naive_bayes,
                      NB_ALPHA_VALUES, X_train, y_train, X_val, y_val)

    models = {
        "Logistic regression": make_logistic_regression(best_c).fit(X_train, y_train),
        "Naive Bayes": make_naive_bayes(best_alpha).fit(X_train, y_train),
    }

    # 4. Evaluate both models on the test set (used only here, once).
    results = {name: evaluate(name, model, X_test, y_test) for name, model in models.items()}
    plot_comparison(results)
    word_lines = top_words(models["Logistic regression"], vectoriser)

    # Save a text summary of all the results for writing up the report.
    with open(os.path.join(OUTPUT_DIR, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best settings: logistic regression C={best_c}, Naive Bayes alpha={best_alpha}\n\n")
        for name, result in results.items():
            f.write(f"=== {name} ===\n")
            f.write(f"Accuracy: {result['accuracy']:.3f}\nMacro F1: {result['macro_f1']:.3f}\n")
            f.write(result["report"] + "\n")
        f.write("Top words per emotion (logistic regression):\n" + "\n".join(word_lines) + "\n")

    # 5. Check the models on our own sentences, then let the user try it.
    test_own_sentences(models, vectoriser)
    print(f"\nCharts and results saved to: {OUTPUT_DIR}")
    interactive_demo(models["Logistic regression"], vectoriser)


if __name__ == "__main__":
    main()
