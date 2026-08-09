from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3, 6],  # 3 samples, 4 features
     [111, 112, 113, 112],
     [200, 3341, 122, 122]]



y = [0, 1, 1]  # classes of each sample
clf.fit(X, y)



print(clf.predict([[114,225, 122, 110], [0, 1, 2, 0]]))  # Output the predictions for new data
print("Hello")

# so next I need to try to write the rds file in list form along the classifiers and train the model.
#I will first test it using the material from one season


# I will try to use random forest classifier to predict win/lose result

# (train on the first 70% of the season chronologically, test on the last 30%.) to avoid leakage
# Steps forward:
# - [ ] Select a single season of unstandardized data
# - [ ] Write 1 for home win, 0 for away win.
# - [ ] Convert the y variables to a list in python. (Idk how to call it from rds?) (or maybe trying another type of data format
# - [ ] Convert the x variables to a list of lists in python. Same as above.
# - [ ] Divide the X and y lists into XTrain yTrain; and XTest, YTest, (70% train, 30% test at random)
# - [ ] Run the random forest classifier on the train data. 
# - [ ] Run the test data and note the percentage. 
# SECOND MAJOR WIN AFTER RUNNING THE VSCODE/PYTHON VERSION
# - [ ] Note the variables with less importance, get rid of them and re-run (see which result is better)


# 3rd win:
# Redo same steps as above, but with all the seasons.

# 4th win:
# Make UI of the prediction model allowing the user to actually input new games. Include it in my portfolio.

# 5th win: (Kind of Extra but would really make this project way more awesome)
# Clean up the code with functions that standardize the raw data coming from the full pbp, matches the momentum variables of the teams and reruns the model to predict. It would be awesome if it auto-updates right after an NFL game in order to use the newer data to predict the next game.

# 6th win: Weight recent games more than past games????  