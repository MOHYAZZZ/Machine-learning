"""
This file implements a decision tree regressor for continuous outputs. `
TreeNode` represents a node, calculating the mean squared error (MSE), identifying the best split, and recursively generating child nodes. 
`RegressionTree` initializes a root `TreeNode`, trains the model, and predicts new instances based on the trained tree.
"""


class TreeNode:
    def __init__(self, examples):
        self.examples = examples
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.label = None

    def compute_mse(self, examples):
        if len(examples) == 0:
            return 0
        labels = [example["bpd"] for example in examples]
        mean = sum(labels) / len(labels)
        mse = sum([(label - mean) ** 2 for label in labels]) / len(labels)
        return mse

    def find_best_split(self):
        min_mse = float('inf')
        best_feature = None
        best_value = None
        features = ["porosity", "gamma", "sonic", "density"]

        for feature in features:
            unique_values = sorted(
                set([example[feature] for example in self.examples]))
            for i in range(1, len(unique_values)):
                split_value = (unique_values[i-1] + unique_values[i]) / 2
                left_examples = [
                    example for example in self.examples if example[feature] <= split_value]
                right_examples = [
                    example for example in self.examples if example[feature] > split_value]
                mse = len(left_examples) / len(self.examples) * self.compute_mse(left_examples) + \
                    len(right_examples) / len(self.examples) * \
                    self.compute_mse(right_examples)
                if mse < min_mse:
                    min_mse = mse
                    best_feature = feature
                    best_value = split_value

        return best_feature, best_value

    def split(self):
        if len(self.examples) <= 1 or self.compute_mse(self.examples) == 0:
            self.label = self.examples[0]["bpd"] if self.examples else None
            return

        self.split_feature, self.split_value = self.find_best_split()

        left_examples = [
            example for example in self.examples if example[self.split_feature] <= self.split_value]
        self.left = TreeNode(left_examples)
        self.left.split()

        right_examples = [
            example for example in self.examples if example[self.split_feature] > self.split_value]
        self.right = TreeNode(right_examples)
        self.right.split()


class RegressionTree:
    def __init__(self, examples):
        self.root = TreeNode(examples)
        self.train()

    def train(self):
        self.root.split()

    def predict(self, example):
        node = self.root
        while node.label is None:
            if example[node.split_feature] <= node.split_value:
                node = node.left
            else:
                node = node.right
        return node.label
