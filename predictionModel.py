from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# Reads data
def read_file(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    return df


# Selects seasons to train
def splitTrain(df):
    data_to_train = df[df["season"] < 2025]
    data_to_train = data_to_train.drop(columns="season")
    data_to_train = data_to_train.drop(columns="spread_line")
    return data_to_train


# Select seasons to test
def splitTest(df):
    data_to_test = df[df["season"] == 2025]
    data_to_test = data_to_test.drop(columns="season")
    data_to_test = data_to_test.drop(columns="spread_line")
    return data_to_test


# Trains RF Classifier
def trainModel(data_to_train):
    clf = RandomForestClassifier(random_state=0)
    y = data_to_train["result"]
    X = data_to_train.drop(columns="result")
    clf.fit(X, y)
    return clf


# Separates the "solutions" before running the test.
def testModel(data_to_test, clf):
    correctY = data_to_test["result"]
    data_to_test = data_to_test.drop(columns="result")
    predictions = clf.predict(data_to_test)
    return [predictions, correctY]


# Prints feature importance if true
def mapImportance(clf, boo):
    if boo == True:
        importance = clf.feature_importances_.tolist()
        variables = clf.feature_names_in_
        mapImp = {}
        for i in range(len(variables)):
            mapImp[variables[i]] = importance[i]
        mapImp = dict(sorted(mapImp.items(), key= lambda item: item[1], reverse= True))
        print(mapImp)


# Gets result metrics
def results(test):
    testY = test[0].tolist()
    correctY = test[1].tolist()
    yNum = len(testY) - 1

    homeWins = 0
    count = 0

    for i in range(yNum):

        if testY[i] == correctY[i]:
            count += 1

        if correctY[i] == 1:
            homeWins += 1

    successRate = count/(yNum + 1)

    print("Success Rate: ", successRate)
    print("Number of home team wins: ", homeWins)
    print("Win rate if it only chose thehome team: ", homeWins/(yNum + 1))



if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    rds_path = base_dir / "data" / "NFL_momentum_data.csv"
    df = read_file(rds_path)

    data_to_train = splitTrain(df)
    clf = trainModel(data_to_train)
    mapImportance(clf, False) #Prints feature importances if True.

    data_to_test = splitTest(df)
    test = testModel(data_to_test, clf)
    
    results(test)


