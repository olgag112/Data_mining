This Python script classifies beer styles based on their features using a Random Forest classifier. It preprocesses the dataset, including converting nominal values to numerical values, removing specific classes, and standardizing features. Key functionalities include:

Classification: A Random Forest model is trained to classify beer styles using features from the dataset.
Data Preprocessing: Nominal-to-numerical conversion (label encoding and one-hot encoding), feature standardization, and filtering out specific classes.
Metrics and Insights: Outputs include accuracy score, classification report, confusion matrix, and insights on most commonly confused classes in the test set.

Files
modified_2.csv: The input dataset.

Key Functions:
findclass: Maps beer sub-genres to main beer style classes.
standardizing: Standardizes a numerical feature in the dataset.
convert_nominal_to_numerical: Converts nominal values into numerical representations (label or one-hot encoding).
most_confused: Highlights the most commonly confused beer styles in the confusion matrix.
