
## The task
https://www.kaggle.com/competitions/heart-disease-prediction-dataquest/
To summarize: I had to develop a machine learning model that would predict heart
disease as a binary classifier

## My Journey:

### First Model of Choice
The first model I decided to use was logistic regression, as it was the simplest
model to base my classifier on. It predicts the probability of a given event
happening given a number of variables. Looking at the dataset this was a good
place to start.

Once I constructed and fitted a plain regression model without any
optimization, this yielded a recall score of 86.42% and a F1 score of 83.83%

### Optimizing the Logistic Regression Model
To optimize the model I did two things: Hyperparameter tuning using GridSearch
and Feature Selection.

For the hyperparameter tuning I used scikit's gridsearch to test what values of
the regularization strength (represented by C), Lasso penalties to add to
prevent overfitting (choosing between l1 and l2, and the maximum iterations to
do). Once those hyperparameters were selected, I did feature selection using
Scikit's SelectFromModel to select features based on weights given
particular requirements.

With these optimizations, the recall score increased to 87.65% and the F1 score
increased to 85.03%. This was about the ceiling I could obtain with the Logistic 
Regression Model. The main issue with it is its boundary based on linear decision 
and inability to learn complex relationships, which would be advantageous for 
this model. Thus, I then constructed my next model based on it: Gradient Boosting

### Gradient Boosting
Gradient Boosting is a strong model as it iteratively evolves, learning from the 
the previous model. Thus, it can train to understand the various characteristics 
that Logistic regression models can't identify, making it a viable model for this 
task. Using XGBM I constructed the next model, and this resulted in a good choice, 
with it showing a 87.07% accuracy without any optimization.

### Optimizing the Gradient Boosting Model
I first attempted feature selection to eliminate any features, but they proved 
to be harmful for the model, as the scores only decreased. Thus, I focused on
hyperparameter tuning instead. To do this, I used the optuna model to dynamically
obtain the best possible parameters of of the model. This proved to be effective
as the recall score increased to 90.12% and F1 score of 87.43%


## Learnings

This was my first foray into applying machine learning in a practical context,
Previously I had studied Linear Regression but learning about binary classification
and the various functioning of the models alongside their strengths and weaknesses
helped improve my understanding of model selection as well as construction and
optimization. This project helped me understand the importance of the hyperparameters
in both the Gradient Boosting and Logistic Regression models as well as how to 
select features in order to better optimize the model.
