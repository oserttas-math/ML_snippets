# Takes a classifier, a grid of hyperparameters as a dictionary, Xtrain, ytrain:

def crossval_optimize(clf, parameters, Xtrain, ytrain, n_folds=10):
    """Prints the best parameter value and returns the best classifier."""

    # Instantiate the grid:
    search = GridSearchCV(clf, param_grid=parameters, cv=n_folds)

    # Fit the model at each grid point:
    search.fit(Xtrain, ytrain)

    print('BEST PARAMETERS:', search.best_params_)
    model = search.best_estimator_

    return model

# Takes a classifier, a grid of hyperparameters as a dictionary, a dataframe as input,
# the target column, and the target value to be assigned the value 1:

def find_and_fit_best(clf, parameters, df, featurenames, targetname, targetval, train_size=0.8, standardize=False):
    """Standardizes features, splits the dataframe, finds the best classifier,
       Returns the best classifier."""

    # Save the features using the featurenames list:
    features = df[featurenames]

    if standardize:
        # Scale the data so that it has 0 mean and
        # is described in units of its standard deviation:
        set_stdrdz = (features - features.mean()) / features.std()
    else:
        set_stdrdz = features

    X = set_stdrdz.values

    # Sets any sample with targetval to the value 1, and all others to 0:
    y = (df[targetname].values == targetval)*1

    # Split the dataframe into 80% training and 20% testing by default:
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size, random_state=5)


    # Train the model on the training set using cross-validation, obtain the best classifier:
    clf = crossval_optimize(clf, parameters, Xtrain, ytrain)

    # Retrain on the entire training set using best classifier returned above:
    clf = clf.fit(Xtrain, ytrain)

    training_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    print("Training Accuracy: {:0.4f}".format(training_accuracy))
    print("Test Accuracy: {:0.4f}".format(test_accuracy))

    return clf
