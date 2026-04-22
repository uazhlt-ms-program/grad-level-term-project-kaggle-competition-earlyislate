import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_logreg_pipeline():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_nb_pipeline():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=False,
                    use_idf=False,
                    norm=None,
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )


def build_linsvc_pipeline():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )


def evaluate_models(train_df: pd.DataFrame, dev_size: float):
    X = train_df["TEXT"].fillna("")
    y = train_df["LABEL"]

    X_train, X_dev, y_train, y_dev = train_test_split(
        X,
        y,
        test_size=dev_size,
        random_state=42,
        stratify=y,
    )

    models = {
        "logreg": build_logreg_pipeline(),
        "naive_bayes": build_nb_pipeline(),
        "linear_svc": build_linsvc_pipeline(),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)
        score = f1_score(y_dev, preds, average="macro")
        results.append((name, score))

        print(f"\n=== {name} ===")
        print(f"Macro F1: {score:.4f}")
        print(classification_report(y_dev, preds, digits=4))

    best_name, best_score = sorted(results, key=lambda x: x[1], reverse=True)[0]

    print("\n=== Summary ===")
    for name, score in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:12s}  macro_f1={score:.4f}")

    best_model = {
        "logreg": build_logreg_pipeline(),
        "naive_bayes": build_nb_pipeline(),
        "linear_svc": build_linsvc_pipeline(),
    }[best_name]

    return best_name, best_score, best_model


def fit_full_and_save_submission(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    output_path: Path,
):
    X_train_full = train_df["TEXT"].fillna("")
    y_train_full = train_df["LABEL"]
    X_test = test_df["TEXT"].fillna("")

    model.fit(X_train_full, y_train_full)
    preds = model.predict(X_test)

    submission = pd.DataFrame(
        {
            "ID": test_df["ID"],
            "LABEL": preds,
        }
    )
    submission.to_csv(output_path, index=False)

    print(f"\nSaved submission file: {output_path}")
    print(submission.head())


def main(train_path: Path, test_path: Path, output_path: Path, dev_size: float):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_train_cols = {"ID", "TEXT", "LABEL"}
    required_test_cols = {"ID", "TEXT"}

    if not required_train_cols.issubset(train_df.columns):
        raise ValueError(f"train.csv must contain columns {required_train_cols}")
    if not required_test_cols.issubset(test_df.columns):
        raise ValueError(f"test.csv must contain columns {required_test_cols}")

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nLabel distribution:")
    print(train_df["LABEL"].value_counts().sort_index())

    best_name, best_score, best_model = evaluate_models(train_df, dev_size=dev_size)
    print(f"\nBest model based on local validation: {best_name} ({best_score:.4f})")

    fit_full_and_save_submission(train_df, test_df, best_model, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--dev-size", type=float, default=0.2)
    args = parser.parse_args()

    main(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        dev_size=args.dev_size,
    )