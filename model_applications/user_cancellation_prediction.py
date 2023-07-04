"""
This script uses PySpark's ML library to implement a logistic regression model.
The purpose of the model is to predict if a user will cancel within the upcoming week, 
based on the user's interaction counts for the past month, week, and day.

Functionality:
- Combines the interaction count features into a single vector column using VectorAssembler.
- Renames the target column 'cancelled_within_week' to 'label'.
- Trains a logistic regression model using specific hyperparameters.
- Uses the trained model to make predictions and returns a DataFrame with user IDs and their associated predictions.

Note: This script assumes the input data has been appropriately preprocessed. Also, it trains and tests 
on the same dataset, which could lead to overfitting in a real-world scenario.
"""

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

def predict_cancellations(user_interaction_df):
    assembler = VectorAssembler(
        inputCols = ["month_interaction_count", "week_interaction_count", "day_interaction_count"], 
        outputCol = "features")

    features_df = assembler.transform(user_interaction_df)
    features_df = features_df.withColumn("label", features_df["cancelled_within_week"])

    lr_model = LogisticRegression(maxIter = 10, threshold = 0.6, elasticNetParam = 1, regParam = 0.1)
    trained_lr_model = lr_model.fit(features_df)

    predictions_df = trained_lr_model.transform(features_df)
    predictions_df = predictions_df.select(['user_id', 'rawPrediction', 'probability', 'prediction'])

    return predictions_df