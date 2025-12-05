import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        # learning_rate (alpha): Controls the size of the step we take during gradient descent.
        self.lr = learning_rate
        
        # lambda_param: The regularization parameter. 
        # It controls the trade-off between having a wide margin vs correctly classifying training points.
        # Higher lambda = harder margin (more strict), Lower lambda = softer margin.
        self.lambda_param = lambda_param
        
        # n_iters: How many times we loop over the entire dataset to update weights.
        self.n_iters = n_iters
        
        # w: The weight vector (normal to the hyperplane).
        self.w = None
        
        # b: The bias (intercept), controlling the hyperplane's position relative to the origin.
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # SVM requires class labels to be -1 and 1 (unlike 0 and 1 in typical binary classification).
        # This allows us to use the sign of the multiplication y * f(x) to determine correctness.
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights with zeros. n_features is the number of dimensions (e.g., 2 for (x,y) points).
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent Loop
        for _ in range(self.n_iters):
            # Iterate over every data point in the dataset
            for idx, x_i in enumerate(X):
                
                # --- THE MARGIN CONDITION ---
                # The linear model is: f(x) = w • x - b
                # We want y_i * (w • x_i - b) >= 1 for a correct classification outside the margin.
                # If this condition is True: The point is correctly classified and safely outside the margin.
                # If False: The point is either misclassified OR inside the margin (too close to boundary).
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # CASE 1: Correct classification & outside margin.
                    # We only update based on the regularization term (lambda * ||w||^2).
                    # The gradient of lambda * ||w||^2 is: 2 * lambda * w.
                    # We subtract this to keep weights small (which maximizes the margin).
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # CASE 2: Misclassified or inside margin.
                    # We update based on Regularization AND the Loss (Hinge Loss).
                    # Gradient w.r.t w: 2 * lambda * w - (y_i * x_i)
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    
                    # Update the bias b.
                    # Note: We subtract because of the specific formula used: (w*x - b).
                    # If the formula was (w*x + b), we would add.
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        # Calculate the hyperplane equation: w • x - b
        approx = np.dot(X, self.w) - self.b
        
        # Returns -1 if the value is negative, +1 if positive.
        # This determines which side of the line the point falls on.
        return np.sign(approx)


# Testing
if __name__ == "__main__":
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # Generate synthetic data: 50 samples, 2 features (x, y coordinates), 2 centers (classes)
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    # Convert 0/1 labels to -1/1 for our SVM
    y = np.where(y == 0, -1, 1)

    # Train the model
    clf = SVM()
    clf.fit(X, y)

    print("Calculated Weights:", clf.w)
    print("Calculated Bias:", clf.b)

    def visualize_svm():
        # --- HYPERPLANE EQUATION HELPER ---
        # The decision boundary is: w0*x + w1*y - b = 0
        # To plot this on a 2D graph, we need y in terms of x.
        # Rearranging the algebra:
        # w1*y = -w0*x + b
        # y = (-w0*x + b) / w1
        # The 'offset' allows us to plot the margins (where equation equals -1 or 1 instead of 0).
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot the actual data points
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        # Get the minimum and maximum x-values from the data to define line length
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        # 1. Decision Boundary (The middle line)
        # Equation: w.x - b = 0 (offset is 0)
        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        # 2. Lower Margin (The "gutter" for class -1)
        # Equation: w.x - b = -1 (offset is -1)
        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        # 3. Upper Margin (The "gutter" for class +1)
        # Equation: w.x - b = 1 (offset is 1)
        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        # Plot the lines on the graph
        # y-- : Yellow dashed line (Decision Boundary)
        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        # k : Black solid lines (Margins)
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        # Set Y-axis limits so the lines don't stretch the graph too far vertically
        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

    visualize_svm()