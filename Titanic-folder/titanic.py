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
    '''
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
    '''

    embarked_survival_counts = pd.crosstab(traindf["Embarked"], traindf["Survived"])
    print(embarked_survival_counts)

    print()

    embarked_class_counts = pd.crosstab(traindf["Embarked"], traindf["Pclass"])
    print(embarked_class_counts)

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
    
    traindf["Title"] = traindf['Name'].str.extract(r', ([A-Za-z]+)\.', expand=False)
    
    traindf['Title'] = traindf['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    traindf['Title'] = traindf['Title'].replace('Mlle', 'Miss')
    traindf['Title'] = traindf['Title'].replace('Ms', 'Miss')
    traindf['Title'] = traindf['Title'].replace('Mme', 'Mrs')

    traindf['Age'] = traindf['Age'].fillna(traindf.groupby('Title')['Age'].transform('mean'))
    traindf['Age'] = traindf['Age'].fillna(traindf['Age'].mean()) 

    mode_embarked = traindf['Embarked'].mode()[0]
    traindf['Embarked'] = traindf['Embarked'].fillna(mode_embarked)

    traindf['CabinLetter'] = traindf['Cabin'].str[0]
    traindf['CabinLetter'] = traindf['CabinLetter'].fillna("N")

    traindf['FamilySize'] = traindf['SibSp'] + traindf['Parch'] + 1
    traindf['IsAlone'] = (traindf['FamilySize'] == 1).astype(int)
    
    sex_map = {"male": 0, "female": 1}
    traindf['Sex'] = traindf['Sex'].map(sex_map)

    traindf['Fare'] = traindf['Fare'].fillna(traindf['Fare'].median())
    
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone',
                'Embarked', 'Title', 'CabinLetter']
    
    y = traindf['Survived']
    X = traindf[features] 
    
    X_processed = pd.get_dummies(X, columns=['Embarked', 'Title', 'CabinLetter'], drop_first=True)
    
    return (X_processed, y)





def get_filtered_X():
    testdf = pd.read_csv('test.csv')
    
    testdf["Title"] = testdf['Name'].str.extract(r', ([A-Za-z]+)\.', expand=False)
    
    testdf['Title'] = testdf['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    testdf['Title'] = testdf['Title'].replace('Mlle', 'Miss')
    testdf['Title'] = testdf['Title'].replace('Ms', 'Miss')
    testdf['Title'] = testdf['Title'].replace('Mme', 'Mrs')

    testdf['Age'] = testdf['Age'].fillna(testdf.groupby('Title')['Age'].transform('mean'))
    testdf['Age'] = testdf['Age'].fillna(testdf['Age'].mean()) 

    mode_embarked = testdf['Embarked'].mode()[0]
    testdf['Embarked'] = testdf['Embarked'].fillna(mode_embarked)

    testdf['CabinLetter'] = testdf['Cabin'].str[0]
    testdf['CabinLetter'] = testdf['CabinLetter'].fillna("N")

    testdf['FamilySize'] = testdf['SibSp'] + testdf['Parch'] + 1
    testdf['IsAlone'] = (testdf['FamilySize'] == 1).astype(int)
    
    sex_map = {"male": 0, "female": 1}
    testdf['Sex'] = testdf['Sex'].map(sex_map)

    testdf['Fare'] = testdf['Fare'].fillna(testdf['Fare'].median())
    
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone',
                'Embarked', 'Title', 'CabinLetter']
    
    X = testdf[features] 
    
    X_processed = pd.get_dummies(X, columns=['Embarked', 'Title', 'CabinLetter'], drop_first=True)
    
    return (X_processed)


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
    for x in range(seedRange):
        score, mse = random_forest_test(seed, 50)
        avgScore += score
        avgMSE += mse
        seed += 1
        print()
    avgScore /= seedRange
    avgMSE /= seedRange
    print(f"Average Random Forest Score over {seedRange} seeds: {avgScore}")
    print(f"Average Random Forest MSE over {seedRange} seeds: {avgMSE}")

#run_random_forest_test()

    
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

'''
def make_submission():
    model = RandomForestClassifier(n_estimators=115, max_depth=15, min_samples_leaf=3)
    X_train, y_train = get_filtered_Xy()
    model.fit(X_train, y_train)
    X_test = get_filtered_X()

    predicitions = model.predict(X_test)

    submission_df = pd.read_csv('test.csv')
    submission_df['Survived'] = predicitions
    submission_df = submission_df[['PassengerId', 'Survived']]
    submission_df.to_csv('submission.csv', index=False)
'''
    
def make_submission():
    model = RandomForestClassifier(n_estimators=115, max_depth=15, min_samples_leaf=3)
    X_train, y_train = get_filtered_Xy()
    X_test = get_filtered_X()

    X_train_aligned, X_test_aligned = X_train.align(X_test, join='outer', axis=1, fill_value=0)
    model.fit(X_train_aligned, y_train)
    predicitions = model.predict(X_test_aligned)

    submission_df = pd.read_csv('test.csv')
    submission_df['Survived'] = predicitions
    submission_df = submission_df[['PassengerId', 'Survived']]
    submission_df.to_csv('submission.csv', index=False)

make_submission()