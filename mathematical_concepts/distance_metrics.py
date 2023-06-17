"""
distance_metrics.py

This module contains a class, Metrics, with methods to calculate different types of distances and similarities
between two numeric vectors or sets. The following metrics are included:

- Euclidean distance
- Manhattan distance
- Cosine similarity
- Jaccard similarity

This module can be used in various applications including but not limited to machine learning algorithms, 
data analysis, and similarity/distance computations between data points.
"""


class Metrics():
    def euclidean_distance(self, X, Y):
        return sum([(x - y) ** 2 for x, y in zip(X, Y)]) ** 0.5

    def manhattan_distance(self, X, Y):
        return sum([abs(x - y) for x, y in zip(X, Y)])

    def cosine_similarity(self, X, Y):
        dot_product = sum([x * y for x, y in zip(X, Y)])
        magnitude_X = sum([x ** 2 for x in X]) ** 0.5
        magnitude_Y = sum([y ** 2 for y in Y]) ** 0.5

        if magnitude_X == 0 or magnitude_Y == 0:
            return 0.0

        return dot_product / (magnitude_X * magnitude_Y)

    def jaccard_similarity(self, X, Y):
        intersection = len(set(X).intersection(set(Y)))
        union = len(set(X).union(set(Y)))
        return intersection / union if union != 0 else 0


def distances_and_similarities(X, Y):
    metrics = Metrics()
    return [
        metrics.euclidean_distance(X, Y),
        metrics.manhattan_distance(X, Y),
        metrics.cosine_similarity(X, Y),
        metrics.jaccard_similarity(X, Y)
    ]
