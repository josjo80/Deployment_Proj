# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.model import train_model, save_model, compute_model_metrics, inference
from ml.data import process_data
import pandas as pd

data = pd.read_csv("../census_mod.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, train_encoder, train_lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=train_encoder, lb=train_lb
)

# Train and save a model.
trained_model = train_model(X_train,y_train)

y_preds = inference(trained_model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
print("Performance on Training data")
print(f"Precision {precision}, Recall {recall}, FBeta {fbeta}")

save_model(trained_model, train_encoder, train_lb)


#Slice testing
def slice_testing(df, model, cat_features, encoder, lb):

    #Iterate through classes
    for cats in cat_features:
        for e in df[cats].unique():
            tmp_df = df[df[cats]==e]
            
            x, y, _, _ = process_data(
                tmp_df, categorical_features=cat_features, 
                label="salary", 
                training=False,
                encoder = train_encoder,
                lb = train_lb
            )

            preds = inference(model,x)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            print(f"Slice testing on {cats} with value {e}")
            print(f"Precision {precision}, Recall {recall}, FBeta {fbeta}")

slice_testing(data, trained_model, cat_features, train_encoder, train_lb)