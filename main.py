import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

random_state = 1

# Get X, y arrays from dataframe
df = pd.read_csv('data.csv')
y = np.array(df['diagnosis'] == 'M').astype(int)
X = np.array(df)[:, 2:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

def fit_and_score_clfs(clfs):
    '''
        clfs: dict of clfs
                key: name of clf
                value: clf object
    '''
    scores = dict()
    for name, clf in clfs.items():
        clf.fit(X_train, y_train)
        scores[name] = clf.score(X_test, y_test)

    return scores

if __name__ == '__main__':

    clfs = {
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'LogisticRegression': LogisticRegression(random_state=random_state),
        'LinearSVC': LinearSVC(random_state=random_state),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=random_state)
    }

    print(fit_and_score_clfs(clfs))