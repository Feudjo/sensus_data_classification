# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The project is implemented by Stephane.
The model is a Decision tree classifier using the defaulthyperparameters in scikit-learn.

## Intended Use
The model should be used to predict weathera person makes 50k a year.

## Training Data
The data was obtained from the UCI Machine Learning Repository.
The training data is made up of 80% of the original data.

## Evaluation Data
The evaluation data is20% of the original data.

## Metrics
The model is evaluated using precision, recall and f_beta score.
## Ethical Considerations


## Caveats and Recommendations
Due to the biased nature of a single tree classifier, a more robust approach
like xgboost, should be considered.
