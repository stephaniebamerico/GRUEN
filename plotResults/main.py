# This is a sample Python script.
import csv
import matplotlib.pyplot as plt

import codecs

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    datasetScores = []
    sampleScores = []
    with open("gruenScoreDataset.csv", newline='', encoding="utf8") as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            try:
                gruen_score = float(row['gruen score'])
                datasetScores.append(gruen_score)
                story = row['story']
            except:
                x = 0
    with open("gruenScoreSamples.csv", newline='', encoding="utf8") as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            try:
                gruen_score = float(row['gruen score'])
                story = row['story']
                sampleScores.append(gruen_score)
            except:
                x = 0
    plt.hist(sampleScores)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
