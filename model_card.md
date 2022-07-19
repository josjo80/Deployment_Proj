# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is based on census data and is used to predict the binary outcome of each individual's salary. The output is `>$50k` or `<=$50k`.
To run, use:
>python3 -m train_model
## Intended Use
This model can be used by organizations to study demographic features with people's income. For instance, this could be used to study income inequality based on a person's background and societal status.
## Training Data
The data was taken from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/census+income. The data was split between 80% training and 20% testing. Categorical features were one-hot encoded and the output label was binarized.
## Evaluation Data
The evaluation was from the same provenance as the training data. The categorical features were one-hot encoded and the output label was binarized.
## Metrics
Precision 0.7927773000859846, Recall 0.5857687420584498, FBeta 0.6737303617099014

## Ethical Considerations
As with all data that includes personal information, care should be taken to ensure that it is deanonymized and that biases have been tested with respect to race. This data does include race and should be tested for biases.
## Caveats and Recommendations
Test for biases in the data using Aequitas. Build a neural network to improve performance of model. Conduct further feature engineering.