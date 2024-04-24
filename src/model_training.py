from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate(classifier_name, X, Y, outlier_fraction):
    """Train and evaluate a given classifier."""
    if classifier_name == "Isolation Forest":
        clf = IsolationForest(
            max_samples=len(X),
            contamination=outlier_fraction,
            random_state=1
        )
        # Use fit() for Isolation Forest
        clf.fit(X)
        y_pred = clf.predict(X)

    elif classifier_name == "Local Outlier Factor":
        clf = LocalOutlierFactor(
            n_neighbors=20,
            contamination=outlier_fraction,
            novelty=False  # Ensures correct mode for outlier detection
        )
        # Use fit_predict() for Local Outlier Factor
        y_pred = clf.fit_predict(X)

    # Convert predictions to binary
    y_pred[y_pred != -1] = 0  # Convert non-outliers to 0
    y_pred[y_pred == -1] = 1  # Convert outliers to 1

    n_errors = (y_pred != Y).sum()

    accuracy = accuracy_score(Y, y_pred)
    report = classification_report(Y, y_pred)

    return {
        "classifier_name": classifier_name,
        "n_errors": n_errors,
        "accuracy": accuracy,
        "report": report
    }





'''def train_and_evaluate(classifier_name, X, Y, outlier_fraction):
    """Train and evaluate a given classifier."""
    if classifier_name == "Isolation Forest":
        clf = IsolationForest(
            max_samples=len(X),
            contamination=outlier_fraction,
            random_state=1
        )
        clf.fit(X)
        y_pred = clf.predict(X)
    elif classifier_name == "Local Outlier Factor":
        clf = LocalOutlierFactor(
            n_neighbors=20,
            contamination=outlier_fraction,
            novelty=False  # For outlier detection
        )
        y_pred = clf.fit_predict(X)

    # Ensure y_pred contains only binary values
    y_pred = (y_pred == -1).astype(int)  # Convert -1 to 1, everything else to 0

    # Check for consistent lengths and valid values
    assert len(y_pred) == len(Y), "Length mismatch between y_pred and Y"
    assert set(y_pred).issubset({0, 1}), "Unexpected values in y_pred"

    accuracy = accuracy_score(Y, y_pred)  # Now safe to compute
    report = classification_report(Y, y_pred)

    return {
        "classifier_name": classifier_name,
        "accuracy": accuracy,
        "report": report
    }'''