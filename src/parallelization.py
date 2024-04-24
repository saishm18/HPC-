import multiprocessing as mp
from joblib import Parallel, delayed

from src.model_training import train_and_evaluate
def run_parallel(classifiers, X, Y, outlier_fraction, n_jobs):
    """Run classifiers in parallel."""
    # Use multiprocessing or joblib for parallelization
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate)(clf, X, Y, outlier_fraction) for clf in classifiers
    )
    return results
