import os
import re
import email
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings

# Suppress sklearn warnings and verbose logs
warnings.filterwarnings("ignore")


class SpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.85, min_df=2)
        self.model = None

    def load_data(self, data_dir="data"):
        """Load and preprocess the SpamAssassin dataset"""
        emails = []
        labels = []

        # Load ham emails
        ham_dir = os.path.join(data_dir, "ham", "easy_ham")
        if os.path.exists(ham_dir):
            for filename in os.listdir(ham_dir):
                if filename.startswith('cmds') or filename == '.DS_Store':
                    continue
                with open(os.path.join(ham_dir, filename), 'r', encoding='latin-1', errors='ignore') as f:
                    emails.append(f.read())
                    labels.append(0)  # 0 for ham

        # Load spam emails
        spam_dir = os.path.join(data_dir, "spam", "spam")
        if os.path.exists(spam_dir):
            for filename in os.listdir(spam_dir):
                if filename.startswith('cmds') or filename == '.DS_Store':
                    continue
                with open(os.path.join(spam_dir, filename), 'r', encoding='latin-1', errors='ignore') as f:
                    emails.append(f.read())
                    labels.append(1)  # 1 for spam

        return emails, labels

    def clean_email(self, text):
        """Clean and preprocess email text"""
        if not isinstance(text, str) or not text.strip():
            return ""

        try:
            msg = email.message_from_string(text)
            body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    try:
                        payload = part.get_payload(decode=True)
                        if not payload:
                            continue
                        payload = payload.decode('latin-1', errors='ignore')
                        if content_type == 'text/plain':
                            body += "\n" + payload
                    except Exception:
                        continue
            else:
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        payload = payload.decode('latin-1', errors='ignore')
                        if msg.get_content_type() == 'text/plain':
                            body = payload
                except Exception:
                    body = text

            if not body.strip():
                body = text

        except Exception:
            body = text

        # Text cleaning
        body = body.lower()
        body = re.sub(r'http\S+|www\S+', ' ', body)
        body = re.sub(r'\S+@\S+', ' ', body)
        body = re.sub(r'\b\d+\b', ' ', body)
        body = re.sub(r'[^a-z\s]', ' ', body)
        body = re.sub(r'\s+', ' ', body).strip()

        return body if body.strip() else ""

    def train(self, X_train, y_train):
        """Train the logistic regression model with hyperparameter tuning"""
        X_train_vec = self.vectorizer.fit_transform(X_train)

        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs'],
            'class_weight': ['balanced', None],
            'penalty': ['l1', 'l2']
        }

        grid_search = GridSearchCV(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=0  # 👈 silence GridSearch logs
        )

        grid_search.fit(X_train_vec, y_train)
        self.model = grid_search.best_estimator_
        return grid_search.cv_results_

    def evaluate(self, X_test, y_test):
        """Evaluate the model silently"""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)
        y_pred_proba = self.model.predict_proba(X_test_vec)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc': roc_auc
        }

    def predict(self, email_text, threshold=0.5):
        """Predict if an email is spam or ham"""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        cleaned_text = self.clean_email(email_text)
        if not cleaned_text.strip():
            return {
                'prediction': 'Invalid email',
                'probability': 0.0,
                'is_spam': False
            }

        text_vec = self.vectorizer.transform([cleaned_text])
        proba = self.model.predict_proba(text_vec)[0]
        spam_prob = proba[1] if self.model.classes_[1] == 1 else proba[0]
        is_spam = spam_prob >= threshold

        return {
            'prediction': 'SPAM' if is_spam else 'HAM',
            'probability': float(spam_prob),
            'is_spam': is_spam
        }


def main():
    classifier = SpamClassifier()
    print("Loading and preprocessing data...")
    emails, labels = classifier.load_data()

    print("Cleaning email texts...")
    cleaned_emails = [classifier.clean_email(email) for email in emails]

    X, y = [], []
    for email, label in zip(cleaned_emails, labels):
        if email.strip():
            X.append(email)
            y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train silently
    classifier.train(X_train, y_train)
    classifier.evaluate(X_test, y_test)

    # Only show the interactive prompt
    print("\n" + "=" * 50)
    print("SPAM CLASSIFIER")
    print("=" * 50)
    print("Enter an email message to classify (press Enter twice to finish):")

    lines = []
    while True:
        line = input()
        if line == '':
            if len(lines) > 0 and lines[-1] == '':
                break
        lines.append(line)

    email_text = '\n'.join(lines).strip()

    if email_text:
        result = classifier.predict(email_text)
        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        print(f"Prediction: {result['prediction']}")
        print(f"Spam Probability: {result['probability'] * 100:.2f}%")
        print("=" * 50)


if __name__ == "__main__":
    main()
