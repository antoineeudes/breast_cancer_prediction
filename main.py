import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np


random_state = 1

# Get X, y arrays from dataframe
df = pd.read_csv('data.csv')
y = np.array(df['diagnosis'] == 'M').astype(int)
X = np.array(df)[:, 2:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

def fit_and_score_clfs(clfs, test_size=0.5):
    '''
        clfs: dict of clfs
                key: name of clf
                value: clf object
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scores = dict()
    for name, clf in clfs.items():
        clf.fit(X_train, y_train)
        scores[name] = clf.score(X_test, y_test)

    return scores

def plot_test_size_influence_over_score(clfs, min_proportion=.1, max_proportion=.9, N=10):
    scores_dict = {name:list() for name in clfs.keys()}
    prop_list = np.linspace(min_proportion, max_proportion, N)

    for test_size in prop_list:
        print(test_size)
        new_scores = fit_and_score_clfs(clfs, test_size=test_size)
        for name, score in scores_dict.items():
            score.append(new_scores[name])


    for name, scores_list in scores_dict.items():
        plt.plot(prop_list, scores_list, label=name)

    plt.legend()
    plt.xlabel('Test proportion')
    plt.ylabel('Score')
    plt.show()

def random_forest_hyper_parameters_random_search(X_train, y_train):
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': np.linspace(10, 200, 10),
        'max_features': list(range(1, 10)),
        'min_samples_leaf': list(range(1, 10)),
        'min_samples_split': list(range(2, 15)),
        'n_estimators': [int(i) for i in np.linspace(100, 1000, 10)]
    }
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2)
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

if __name__ == '__main__':

    # clfs = {
    #     'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=random_state),
    #     'LogisticRegression': LogisticRegression(solver='lbfgs', random_state=random_state),
    #     'LinearSVC': LinearSVC(max_iter= 1000, random_state=random_state),
    #     'GradientBoostingClassifier': GradientBoostingClassifier(random_state=random_state)
    # }

    # print(fit_and_score_clfs(clfs))


    # plot_test_size_influence_over_score(clfs, N=30)
    random_forest_hyper_parameters_random_search(X_train, y_train)