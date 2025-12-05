import numpy as np
from collections import Counter

def entropy(y):
    """
    Calculates the entropy of a label distribution.
    Entropy is a measure of impurity or disorder. High entropy = mixed labels. Low entropy = pure labels.
    Formula: H(S) = - sum(p_i * log2(p_i))
    """
    # Count occurrences of each class label
    hist = np.bincount(y)
    # Calculate probability p_i = count / total_samples
    ps = hist / len(y)
    
    # Apply the entropy formula
    # We filter for p > 0 to avoid computing log2(0) which is undefined
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    """
    Represents a single node in the decision tree.
    It can be a decision node (with a split feature/threshold) or a leaf node (with a value).
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature      # Index of the feature used to split this node
        self.threshold = threshold  # The value of the feature to split on
        self.left = left            # Child node for data points <= threshold
        self.right = right          # Child node for data points > threshold
        self.value = value          # The class label (if this is a leaf node)

    def is_leaf_node(self):
        # A node is a leaf if it holds a specific class prediction value
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split # Stopping criteria: Min samples required to split
        self.max_depth = max_depth                 # Stopping criteria: Max depth to prevent overfitting
        self.n_feats = n_feats                     # Feature subset size (Random Forest style)
        self.root = None

    def fit(self, X, y):
        # Determine how many features to check for splits (default is all features)
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        # Start the recursive tree growth from the root
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        # Traverse the trained tree for every sample in the input X
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # --- STOPPING CRITERIA ---
        # 1. Max depth reached?
        # 2. Node is pure (only 1 class label present)?
        # 3. Too few samples to justify another split?
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            # Create a Leaf Node with the most common class label
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Select a random subset of features to optimize (helps with variance if building a forest)
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # --- GREEDY SEARCH ---
        # Find the single best feature and threshold that maximizes Information Gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # Generate indices for left and right children based on that best split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        
        # Recursively grow the left and right subtrees
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        # Return the Decision Node storing the split info and children
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        # Loop over selected features
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx] # Get all values for this feature
            thresholds = np.unique(X_column) # Potential thresholds are the unique values
            
            # Loop over all possible thresholds for this feature
            for threshold in thresholds:
                # Calculate how good this split is
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # 1. Calculate parent entropy (impurity before split)
        parent_entropy = entropy(y)

        # 2. Simulate the split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0 # No information gain if a split doesn't actually separate data

        # 3. Calculate weighted average entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        
        # Weighted Avg Formula: (N_left/N_total)*E_left + (N_right/N_total)*E_right
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # 4. Information Gain = Entropy(Parent) - WeightedEntropy(Children)
        # We want to maximize this (maximize the reduction in impurity)
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        # Returns indices of samples that go left (<= threshold) vs right (> threshold)
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        # Base case: We reached a leaf, return the prediction
        if node.is_leaf_node():
            return node.value

        # Decision: Go left or right based on feature value vs threshold
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        # Helper to find majority class in a set of labels
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)