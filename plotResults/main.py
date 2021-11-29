import csv
import matplotlib.pyplot as plt

import codecs


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def plotScoreBySampleNumber(allScores):
    orderedGruen = []
    sampleNumbers = []
    for index, sampleScore in enumerate(allScores):
        if index % 10 == 0 and sampleScore[1] != 0:
            sampleNumbers.append(sampleScore[0])
            orderedGruen.append(sampleScore[1])

    plt.figure().clear()

    plt.scatter(sampleNumbers, orderedGruen)
    plt.title('Gruen Scores for Generated Samples')
    outfile = 'gruenSampleScore' + '.pdf'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')


def plotDatasetScore(allScores):
    orderedGruen = []
    sampleNumbers = []

    for index, sampleScore in enumerate(allScores):
        if index % 10 == 0 and sampleScore[1] != 0:
            sampleNumbers.append(index)
            orderedGruen.append(sampleScore[1])

    plt.figure().clear()

    plt.scatter(sampleNumbers, orderedGruen)
    plt.title('Gruen Scores for Dataset')
    outfile = 'gruenDatasetScore' + '.pdf'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')


def boxplotSampleResults(samplesAllTheScores, datasetAllTheScores):
    orderedGruenSample = []

    for index, sampleScore in enumerate(samplesAllTheScores):
        if index % 10 == 0 and sampleScore[1] != 0:
            orderedGruenSample.append(sampleScore[1])
    orderedGruenDataset = []

    for index, sampleScore in enumerate(datasetAllTheScores):
        if index % 10 == 0 and sampleScore[1] != 0:
            orderedGruenDataset.append(sampleScore[1])

    plt.figure().clear()

    fig1, ax1 = plt.subplots()
    ax1.set_title('Gruen Scores')
    ax1.boxplot([orderedGruenSample, orderedGruenDataset])
    # plt.xticks(ticks=xticks, labels=xticksLabels, rotation=45)
    fig1.subplots_adjust(
        top=0.9,
        bottom=0.2,
        left=0.08,
        right=0.981,
        hspace=0.2,
        wspace=0.2
    )
    legend = ["Samples", "Dataset"]
    xTicks = []
    for i in range(1, 3):
        xTicks.append(i)
    xtick = xTicks
    plt.xticks(xtick, legend, rotation=45)

    plt.savefig("boxplotGruenScoreComparison" + ".pdf")

    # plt.scatter(sampleNumbers, orderedGruen)
    # plt.title('Gruen Scores for Dataset')
    # outfile = 'gruenDatasetScore' + '.pdf'
    # plt.savefig(outfile, dpi=300, bbox_inches='tight')


def sortSamples(samplesWithScoreList):
    return samplesWithScoreList[0]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    datasetScores = []
    sampleScores = []
    datasetGrammaticality_score = []
    sampleGrammaticality_score = []
    datasetRedundancy_score = []
    sampleRedundancy_score = []
    datasetFocus_score = []
    sampleFocus_score = []
    sampleNumber = []
    sampleAllScores = []
    datasetAllScores = []
    with open("gruenScoreDatasetAllScores.csv", newline='', encoding="utf8") as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            try:
                gruen_score = float(row['gruen score'])
                grammaticality_score = float(row['grammaticality_score'])
                redundancy_score = float(row['redundancy_score'])
                focus_score = float(row['focus_score'])
                datasetScores.append(gruen_score)
                datasetGrammaticality_score.append(grammaticality_score)
                datasetRedundancy_score.append(redundancy_score)
                datasetFocus_score.append(focus_score)
                story = row['story']
                fileName = row['fileName']
                datasetAllScores.append(
                    [fileName, gruen_score, grammaticality_score, redundancy_score, focus_score])
            except:
                x = 0
    with open("gruenScoreSamplesAllScores.csv", newline='', encoding="utf8") as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            try:
                gruen_score = float(row['gruen score'])
                grammaticality_score = float(row['grammaticality_score'])
                redundancy_score = float(row['redundancy_score'])
                focus_score = float(row['focus_score'])
                sampleScores.append(gruen_score)
                sampleGrammaticality_score.append(grammaticality_score)
                sampleRedundancy_score.append(redundancy_score)
                sampleFocus_score.append(focus_score)
                fileName = row['fileName']
                fileName = fileName.replace('samples-', '')
                sampleNumber.append(int(fileName))
                sampleAllScores.append(
                    [int(fileName), gruen_score, grammaticality_score, redundancy_score, focus_score])
            except:
                print('error')
                x = 0

    sampleAllScores.sort(key=sortSamples)
    plotScoreBySampleNumber(sampleAllScores)
    boxplotSampleResults(sampleAllScores, datasetAllScores)
    plotDatasetScore(datasetAllScores)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
