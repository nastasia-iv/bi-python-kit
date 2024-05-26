import numpy as np

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ThreadPoolExecutor

SEED = 42

class RandomForestClassifierCustom(BaseEstimator):
    """
    Class for custom implementation of a Random Forest classifier.
    """
    def __init__(
        self, n_estimators: int = 10, max_depth: int = None,
        max_features: int = None, random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.n_features = None
        self.n_samples = None
        self.classes = None

        self.trees = []
        self.feat_ids_by_tree = []

    def _fit_tree(self, tree_id: int, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        """
        Training a single tree in a Random Forest.

        Arguments:
            tree_id (int): Tree number.
            X (numpy.ndarray): Feature matrix for training.
            y (numpy.ndarray): Target variable for training.

        Returns:
            DecisionTreeClassifier: Trained decision tree.
        """
        tree_seed = SEED + tree_id
        np.random.seed(tree_seed)

        # Select random features for a tree
        feature_ids = np.random.choice(self.n_features, self.max_features, replace=False)
        self.feat_ids_by_tree.append(feature_ids)

        # Select random samples to train a tree
        sample_ids = np.random.choice(self.n_samples, self.n_samples, replace=True)

        # Train a decision tree with selected features and samples
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth, max_features=self.max_features, random_state=tree_seed
        )
        tree.fit(X[sample_ids][:, feature_ids], y[sample_ids])
        return tree

    def fit(self, X: np.ndarray, y: np.ndarray, n_jobs: int = 1) -> 'RandomForestClassifierCustom':
        """
        Training Random Forest model.

        Arguments:
            X (numpy.ndarray): Feature matrix for training.
            y (numpy.ndarray): Target variable for training.
            n_jobs (int): Number of threads for parallel training of trees.

        Returns:
            RandomForestClassifierCustom: Trained instance of RandomForestClassifierCustom.
        """
        self.classes = sorted(np.unique(y))
        self.n_samples, self.n_features = X.shape

        # Parallel training of all trees
        with ThreadPoolExecutor(n_jobs) as pool:
            self.trees = list(pool.map(lambda tree_id: self._fit_tree(tree_id, X, y), range(self.n_estimators)))

        return self

    def predict_proba(self, X: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        """
        Predicting class probabilities for new data.

        Arguments:
            X (numpy.ndarray): Feature matrix for prediction.
            n_jobs (int): Number of threads for parallel probability prediction.

        Returns:
            numpy.ndarray: A matrix of class probabilities for each sample.
        """
        probas = np.zeros((X.shape[0], len(self.classes)))

        def predict_proba_tree(tree: DecisionTreeClassifier, feature_ids: np.ndarray, X: np.ndarray) -> np.ndarray:
            """
            Predicting class probabilities for new data using a single tree.

            Arguments:
                tree (DecisionTreeClassifier): Trained decision tree.
                feature_ids (numpy.ndarray): Selected features for the given tree.
                X (numpy.ndarray): Feature matrix for prediction.

            Returns:
                numpy.ndarray: A matrix of class probabilities for each sample.
            """
            return tree.predict_proba(X[:, feature_ids])

        # Parallel probability prediction for all trees
        with ThreadPoolExecutor(n_jobs) as pool:
            results = list(pool.map(predict_proba_tree, self.trees, self.feat_ids_by_tree, [X]*len(self.trees)))

        for tree_probas in results:
            probas += tree_probas

        probas /= self.n_estimators

        return probas

    def predict(self, X: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        """
        Predicting classes for new data.

        Arguments:
            X (numpy.ndarray): Feature matrix for prediction.
            n_jobs (int): Number of threads for parallel class prediction.

        Returns:
            numpy.ndarray: A vector of predicted classes for each sample.
        """
        probas = self.predict_proba(X, n_jobs)
        predictions = np.argmax(probas, axis=1)

        return predictions
