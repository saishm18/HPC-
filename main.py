from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model_training import train_and_evaluate
from src.parallelization import run_parallel
from datetime import datetime
import time

startTime = datetime.now()
# Load the dataset
data = load_data("data/creditcard.csv")

# Preprocess the data
data = preprocess_data(data)

# Get features and target variable
columns = [col for col in data.columns if col != "Class"]
X = data[columns]
Y = data["Class"]

# Outlier fraction
Fraud = data[data["Class"] == 1]
Valid = data[data["Class"] == 0]
outlier_fraction = len(Fraud) / float(len(Valid))

# Classifiers to run in parallel
classifiers = ["Isolation Forest", "Local Outlier Factor"]

# Run the classifiers in parallel
results = run_parallel(classifiers, X, Y, outlier_fraction, n_jobs=1)

# Display the results
for result in results:
    print(f"Classifier: {result['classifier_name']}")
    print(f"Errors: {result['n_errors']}")
    print(f"Accuracy: {result['accuracy']:.2%}")
    print(f"Classification Report:\n{result['report']}")

print("execution time = ",datetime.now() - startTime)  