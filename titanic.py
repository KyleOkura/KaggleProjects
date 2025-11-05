import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def explore():
    testdf = pd.read_csv('test.csv')
    traindf = pd.read_csv('train.csv')

    traindf["Title"] = traindf['Name'].str.extract(r', ([A-Za-z]+)\.', expand=False)

    title_survival_counts = pd.crosstab(traindf["Title"], traindf["Survived"])
    missing_age_count = traindf['Age'].isnull().sum()
    print(title_survival_counts)
    print()
    print(f'# entiries missing age count: {missing_age_count}')
    print(f'total entries: {len(traindf)}')
    print()

    print(f'Mean age by title: {traindf['Age'].groupby(traindf['Title']).mean()}')
    print()
    print(f'Median age by title: {traindf['Age'].groupby(traindf['Title']).median()}')
    print(traindf['Age'].groupby(traindf['Title']).mean()['Mr'])
    #print(traindf["Title"].value_counts())

#explore()

'''
def get_filtered_Xy():
    traindf = pd.read_csv('train.csv')
    sex_map = {"male": 0, "female": 1}
    traindf['Sex'] = traindf['Sex'].map(sex_map)
    mean_age = int(traindf['Age'].mean())
    traindf['Age'] = traindf["Age"].fillna(mean_age)

    mode_embarked = traindf['Embarked'].mode()[0]
    traindf['Embarked'] = traindf['Embarked'].fillna(mode_embarked)
    embarked_map = {"C": 0, "Q": 1, "S": 2}
    traindf['Embarked'] = traindf['Embarked'].map(embarked_map)

    #X = traindf[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    X = traindf[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
    y = traindf['Survived']

    return (X, y)

'''

def get_filtered_Xy():
    traindf = pd.read_csv('train.csv')
    sex_map = {"male": 0, "female": 1}
    traindf['Sex'] = traindf['Sex'].map(sex_map)

    traindf["Title"] = traindf['Name'].str.extract(r', ([A-Za-z]+)\.', expand=False)
    traindf['Age'] = traindf['Age'].fillna(traindf.groupby('Title')['Age'].transform('mean'))

    mean_age = int(traindf['Age'].mean())
    traindf['Age'] = traindf["Age"].fillna(mean_age)

    mode_embarked = traindf['Embarked'].mode()[0]
    traindf['Embarked'] = traindf['Embarked'].fillna(mode_embarked)
    embarked_map = {"C": 0, "Q": 1, "S": 2}
    traindf['Embarked'] = traindf['Embarked'].map(embarked_map)

    #X = traindf[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    X = traindf[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
    y = traindf['Survived']

    return (X, y)





def get_filtered_X():
    traindf = pd.read_csv('test.csv')
    sex_map = {"male": 0, "female": 1}
    traindf['Sex'] = traindf['Sex'].map(sex_map)

    traindf["Title"] = traindf['Name'].str.extract(r', ([A-Za-z]+)\.', expand=False)
    traindf['Age'] = traindf['Age'].fillna(traindf.groupby('Title')['Age'].transform('mean'))

    mean_age = int(traindf['Age'].mean())
    traindf['Age'] = traindf["Age"].fillna(mean_age)

    mode_embarked = traindf['Embarked'].mode()[0]
    traindf['Embarked'] = traindf['Embarked'].fillna(mode_embarked)
    embarked_map = {"C": 0, "Q": 1, "S": 2}
    traindf['Embarked'] = traindf['Embarked'].map(embarked_map)

    #X = traindf[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    X = traindf[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

    return (X)


def linear_reg_test(seed):
    """
    Nan ages filled with mean age
    ticket dropped
    Cabin dropped
    Embarked dropped

    linear regression model
    """
    X, y = get_filtered_Xy()

    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model.fit(X_train, y_train)

    float_predictions = model.predict(X_test)
    predictions = (float_predictions >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, predictions)

    #print(f"linear reg test accurancy: {accuracy}")
    return accuracy


def decision_tree_test(seed, depth=4):
    X, y = get_filtered_Xy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    clf = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    clf.fit(X_train, y_train)

    '''
    plt.figure(figsize=(15,10))
    plot_tree(clf)
    plt.show()
    '''

    accuracy = clf.score(X_test, y_test)
    #print(f"decision tree test accurancy: {accuracy}")
    return accuracy


def run_decision_tree_test():
    seed = 1

    for x in range(100):
        decision_tree_test(seed)
        seed += 1



def random_forest_test(seed, num_leaves):
    X, y = get_filtered_Xy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    clf = RandomForestClassifier(n_estimators=num_leaves, random_state=seed)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    mse = mean_absolute_error(y_test, clf.predict(X_test))
    print(f"random forest test accurancy: {accuracy}")
    print(f"random forest test MSE: {mse}")
    return accuracy, mse



def run_random_forest_test():
    seed = 1
    #n_estimators_list = [1, 5, 10, 20, 50, 100, 150, 200, 300, 500]
    avgScore = 0
    avgMSE = 0
    seedRange = 50
    for x in range(50):
        score, mse = random_forest_test(seed, 50)
        avgScore += score
        avgMSE += mse
        seed += 1
        print()
    avgScore /= seedRange
    avgMSE /= seedRange
    print(f"Average Random Forest Score over {seedRange} seeds: {avgScore}")
    print(f"Average Random Forest MSE over {seedRange} seeds: {avgMSE}")

run_random_forest_test()

    
#explore()
def run_tests():
    seed = 1
    linear_reg_count = 0
    decision_tree_count = 0
    tie_count = 0

    linear_reg_avg = 0
    decision_tree_avg = 0

    num_seeds = 100

    for x in range(num_seeds):
        linear_reg_test_score = linear_reg_test(seed)
        decision_tree_test_score = decision_tree_test(seed)

        linear_reg_avg += linear_reg_test_score
        decision_tree_avg += decision_tree_test_score


        '''
        print(seed)
        print(f"linear reg score: {linear_reg_test_score}")
        print(f"decision tree score: {decision_tree_test_score}")
        print()
        '''

        if linear_reg_test_score > decision_tree_test_score:
            linear_reg_count += 1
        elif decision_tree_test_score > linear_reg_test_score:
            decision_tree_count += 1
        elif decision_tree_test_score == linear_reg_test_score:
            tie_count += 1
        else:
            print("scoring error")


        seed += 1

    linear_reg_avg /= num_seeds
    decision_tree_avg /= num_seeds


    print(f"linear reg count: {linear_reg_count}")
    print(f"decision tree count: {decision_tree_count}")
    print(f"ties: {tie_count}")

    print(f"linear reg avg score: {round(linear_reg_avg,3)}")
    print(f"decision tree avg score: {round(decision_tree_avg,3)}")


def make_submission():
    model = RandomForestClassifier(n_estimators=50)
    X_train, y_train = get_filtered_Xy()
    model.fit(X_train, y_train)

    X_test = get_filtered_X()
    predicitions = model.predict(X_test)

    submission_df = pd.read_csv('test.csv')
    submission_df['Survived'] = predicitions
    submission_df = submission_df[['PassengerId', 'Survived']]
    submission_df.to_csv('submission.csv', index=False)
    

#make_submission()