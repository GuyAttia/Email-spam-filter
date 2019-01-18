from sklearn import tree
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

np.random.seed(3)
columns_names = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
                 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
                 'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
                 'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
                 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
                 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
                 'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
                 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
                 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
                 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
                 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
                 'capital_run_length_longest', 'capital_run_length_total', 'spam']


def get_data():
    """
    Split the data into features(x) and target(y)
    :return: Features and target
    """
    df = pd.read_csv('spambase.data.txt', header=None, names=columns_names)
    x = df.iloc[:, :-1]
    x = change_to_categorical(x)
    y = df.iloc[:, -1]
    return x, y


def change_to_categorical(x):
    for i, col in enumerate(x):
        if i < 54:
            x.loc[(x[col]>0) & (x[col]<1), col] = 0.5
            x.loc[(x[col]>=1) & (x[col]<3), col] = 3
            x.loc[(x[col]>=3) & (x[col]<5), col] = 5
            x.loc[x[col]>=5, col] = 10
        elif i == 54:
            x.loc[(x[col]>0) & (x[col]<1), col] = 0
            x.loc[(x[col]>=1) & (x[col]<1.5), col] = 1.5
            x.loc[(x[col]>=1.5) & (x[col]<=2), col] = 2
            x.loc[(x[col]>2) & (x[col]<=4), col] = 4
            x.loc[x[col]>4, col] = 10
        elif i == 55:
            x.loc[x[col]<3, col] = 1
            x.loc[(x[col]>=6) & (x[col]<=7), col] = 7
            x.loc[(x[col]>7) & (x[col]<=10), col] = 10
            x.loc[(x[col]>=13) & (x[col]<=15), col] = 15
            x.loc[(x[col]>15) & (x[col]<=25), col] = 25
            x.loc[(x[col]>25) & (x[col]<=50), col] = 50
            x.loc[x[col]>50, col] = 100
        elif i == 56:
            x.loc[x[col]<=6, col] = 6
            x.loc[(x[col]>6) & (x[col]<=11), col] = 11
            x.loc[(x[col]>11) & (x[col]<=16), col] = 16
            x.loc[(x[col]>16) & (x[col]<=22), col] = 22
            x.loc[(x[col]>22) & (x[col]<=30), col] = 30
            x.loc[(x[col]>30) & (x[col]<=50), col] = 50
            x.loc[x[col]>50, col] = 80
    return x


def buildTree(k):
    x, y = get_data()
    skb = SelectKBest(chi2, k=10).fit(x, y)
    x_new = skb.transform(x) # Prune vertices
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=1-k)  # Split to train and test
    k_cross_folds = int(input('Insert the number of wanted folds (>=2) for the k-cross validation\n'))
    clf = treeError(k_cross_folds, x_train, y_train)
    y_preds = clf.predict(x_test)
    test_error = accuracy_score(y_preds, y_test)
    print('Test accuracy score is {}'.format(test_error))
    print('Printing the Tree:\n{0}\n{1}'.format(clf, tree.export_graphviz(clf, out_file=None)))
    tree.export_graphviz(clf, out_file='tree.dot')
    print('The tree saved also in tree.dot file')
    return clf, skb


def treeError(k, x, y):
    clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5)
    kf = KFold(n_splits=k)
    scores = []

    for train_index, valid_index in kf.split(x):    # k-cross-validation
        x_train, x_valid = x[train_index], x[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        clf = clf.fit(x_train, y_train)
        y_preds = clf.predict(x_valid)
        valid_error = accuracy_score(y_preds, y_valid)
        scores.append(valid_error)

    cv_score = np.mean(scores)
    print('The k-cross validation mean accuracy score is {}'.format(cv_score))
    return clf


def isThisSpam(x):
    x_new = skb.transform(x)
    y_pred = clf.predict(x_new)
    print('Prediction for the received email is\n{}'.format(y_pred[0]))
    return y_pred[0]


if __name__ == '__main__':
    k = input('Insert the train proportion in [0,1] range\n')
    clf, skb = buildTree(float(k))
    print('Example for input email')
    x = [[0,0.07,0.14,0,0.14,0.07,0,0,0,0,0,1.34,0.07,0.14,0,0,0.63,0,0.14,0,0,0,0.07,0,3.03,0,0,0,0,0,0,0,0,0,0,0.07,0.21,0,0,0,0,0,0,0,0,0,0,0,0.084,0.177,0,0,0,0,2.25,26,855]]
    y = isThisSpam(x)
