
## The task
https://www.kaggle.com/competitions/heart-disease-prediction-dataquest/
To summarize: I had to develop a machine learning model that would predict heart
disease as a binary classifier

## My Journey:

### First Model of Choice The first model I decided to use was logistic
regression, as it was the simplest model to base my classifier on. It predicts
the probability of a given event happening given a number of variables. Looking
at the dataset this was a good place to start.

Once I constructed and fitted a vanilla regression model without any
optimization, this yielded an accuracy of 81.63%, with a F1 score of 83.83%. 

### Optimizing the Logistic Regression Model
To optimize the model I did two things: Hyperparameter tuning using GridSearch 
and Feature Selection.

For the hyperparameter tuning I used scikit's gridsearch to test what values of
the regularization strength (represented by C), Lasso penalties to add to
prevent overfitting (choosing between l1 and l2, and the maximum iterations to
do). Once those hyperparameters were selected, I did feature selection using
Scikit's SelectFromModel to select features based on weights given
particular requirements.

With these optimizations, the accuracy jumped to 82.99% with a F1-score of
85.03%. This was about the ceiling I could obtain with the Logistic Regression
Model. The main issue with it is its boundary based on linear decision and
inability to learn complex relationships, which would be advantageous for this
model. Thus, I then constructed my next model based on it: Gradient Boosting

### Gradient Boosting Gradient Boosting is a strong model as it iteratively
evolves, learning from the mistakes of the previous model. Thus, it can train
to understand the various characteristics that Logistic regression models can't
identify, making it a viable model for this task. Using XGBM I constructed the
next model, and this resulted in a good choice, with it showing a 87.07%
accuracy without any optimization.

