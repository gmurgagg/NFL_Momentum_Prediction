from pathlib import Path
import subprocess
import tempfile
from sklearn.ensemble import RandomForestClassifier

import pandas as pd


def read_rds_file(path):
    """Load an .rds file into a Python pandas DataFrame."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RDS file not found: {path}")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        csv_path = Path(tmp.name)

    try:
        r_script = f"""
        x <- readRDS('{path}')
        if (!is.data.frame(x)) {{
          stop('The RDS file does not contain a data.frame.')
        }}
        write.csv(x, file = '{csv_path}', row.names = FALSE)
        """

        subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            check=True,
        )

        df = pd.read_csv(csv_path)
        return df
    finally:
        if csv_path.exists():
            csv_path.unlink(missing_ok=True)


def splitTrain(df):
    data_to_train = df[df["season"] < 2025]
    data_to_train = data_to_train.drop(columns="season")
    data_to_train = data_to_train.drop(columns="spread_line")

    return data_to_train

def splitTest(df):
    data_to_test = df[df["season"] == 2025]
    data_to_test = data_to_test.drop(columns="season")
    data_to_test = data_to_test.drop(columns="spread_line")

    return data_to_test


#I should delete the time variables, or/and any other variables that add noise to the modeling. I need to check how to find that noise.

def trainModel(data_to_train):
    clf = RandomForestClassifier(random_state=0)
    y = data_to_train["result"].tolist()
    X = data_to_train.drop(columns="result") #should I also convert this to list?
    clf.fit(X, y)
    return clf


def testModel(data_to_test, clf):
    correctY = data_to_test["result"].tolist()
    data_to_test = data_to_test.drop(columns="result")

    predictions = clf.predict(data_to_test)
    return [predictions, correctY]


def results(test):
    testY = test[0].tolist()
    correctY = test[1]
    yNum = len(testY) - 1

    chooseHome = 0
    count = 0

    for i in range(yNum):

        if testY[i] == correctY[i]:
            count += 1

        if correctY[i] == 1:
            chooseHome += 1

    successRate = count/(yNum + 1)

    print("Success Rate: ", successRate)
    print("Number of games it chose home: ", chooseHome)
    print("choose Home rate: ", chooseHome/(yNum + 1))

        


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    rds_path = base_dir / "rdsFiles" / "randomClfMerge_1.rds"

    df = read_rds_file(rds_path)
    data_to_train = splitTrain(df)
    clf = trainModel(data_to_train)

    data_to_test = splitTest(df)
    test = testModel(data_to_test, clf)
    results(test)


