import csv
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import statistics

import codecs

GRAMATICALITY = 2
REDUNCANCY = 3
FOCUS = 4


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def plotScoreHistogramSample(allScores):
    orderedGruen = []
    sampleNumbers = []
    for index, sampleScore in enumerate(allScores):
        sampleNumbers.append(sampleScore[0])
        orderedGruen.append(sampleScore[1])

    plt.figure().clear()
    plt.close('all')

    plt.hist(orderedGruen, weights=np.ones(len(orderedGruen)) / len(orderedGruen))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Gruen Scores Histogram for Generated Samples')
    outfile = 'gruenSampleScoreHist' + '.pdf'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')


def plotScoreHistogramDataset(allScores):
    orderedGruen = []
    sampleNumbers = []
    for index, sampleScore in enumerate(allScores):
        sampleNumbers.append(sampleScore[0])
        orderedGruen.append(sampleScore[1])

    plt.figure().clear()
    plt.close('all')

    plt.hist(orderedGruen, weights=np.ones(len(orderedGruen)) / len(orderedGruen))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.title('Gruen Scores Histogram for Dataset')
    outfile = 'gruenDatasetScoreHist' + '.pdf'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')


def plotTrend(allScores, title, filename, plotZero=False):
    orderedGruen = []
    sampleNumbers = []

    for index, sampleScore in enumerate(allScores):
        if index % 10 == 0 and (plotZero or sampleScore[1] != 0):
            sampleNumbers.append(index)
            orderedGruen.append(sampleScore[1])

    plt.figure().clear()
    plt.close('all')
    x = sampleNumbers
    y = orderedGruen
    plt.scatter(sampleNumbers, orderedGruen)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")

    plt.title(title)
    outfile = filename + '.pdf'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')


def plotScore(allScores, title, outfile, plotZero=False):
    orderedGruen = []
    sampleNumbers = []

    for index, sampleScore in enumerate(allScores):
        if index % 10 == 0 and (plotZero or sampleScore[1] != 0):
            sampleNumbers.append(index)
            orderedGruen.append(sampleScore[1])

    plt.figure().clear()
    plt.close('all')

    plt.scatter(sampleNumbers, orderedGruen)
    plt.title(title)
    outfile = outfile + '.pdf'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')


def boxplotSampleResults(samplesAllTheScores, datasetAllTheScores, plotZero=False):
    orderedGruenSample = []

    for index, sampleScore in enumerate(samplesAllTheScores):
        if index % 10 == 0 and (plotZero or sampleScore[1] != 0):
            orderedGruenSample.append(sampleScore[1])
    orderedGruenDataset = []

    for index, sampleScore in enumerate(datasetAllTheScores):
        if index % 10 == 0 and (plotZero or sampleScore[1] != 0):
            orderedGruenDataset.append(sampleScore[1])

    orderedGruenSampleWithZero = []

    for index, sampleScore in enumerate(samplesAllTheScores):
        if index % 10 == 0:
            orderedGruenSampleWithZero.append(sampleScore[1])
    orderedGruenDatasetWithZero = []

    for index, sampleScore in enumerate(datasetAllTheScores):
        if index % 10 == 0:
            orderedGruenDatasetWithZero.append(sampleScore[1])

    plt.figure().clear()
    plt.close('all')

    fig1, ax1 = plt.subplots()
    ax1.set_title('Gruen Scores')
    ax1.boxplot([orderedGruenSample, orderedGruenSampleWithZero, orderedGruenDataset, orderedGruenDatasetWithZero])
    # plt.xticks(ticks=xticks, labels=xticksLabels, rotation=45)
    fig1.subplots_adjust(
        top=0.9,
        bottom=0.2,
        left=0.08,
        right=0.981,
        hspace=0.2,
        wspace=0.2
    )
    legend = ["Amostras", "Amostras\n com Zeros", "Base de Dados", "Base de Dados\n com zeros"]
    xTicks = []
    for i in range(1, 5):
        xTicks.append(i)
    xtick = xTicks
    plt.xticks(xtick, legend, rotation=45)

    plt.savefig("boxplotGruenScoreComparison" + ".pdf")

    # plt.scatter(sampleNumbers, orderedGruen)
    # plt.title('Gruen Scores for Dataset')
    # outfile = 'gruenDatasetScore' + '.pdf'
    # plt.savefig(outfile, dpi=300, bbox_inches='tight')


def sortByScore(samplesWithScoreList):
    return samplesWithScoreList[1]


def sortSamples(samplesWithScoreList):
    return samplesWithScoreList[0]


def boxplotTwoInstances(samplesAllTheScores, datasetAllTheScores, title, filename):
    plt.figure().clear()
    plt.close('all')

    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.boxplot([samplesAllTheScores, datasetAllTheScores])
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

    plt.savefig(filename + ".pdf")


def plotLines(sampleScores, datasetScores, title, outfile, plotZero=False):
    ticks = list(range(0, 200))
    sampleNumbers = random.sample(sampleScores, 1000)
    sampleNumbers.sort()
    datasetNumbers = random.sample(datasetScores, 1000)
    datasetNumbers.sort()

    # for index, sampleScore in enumerate(sampleScores, datasetScores):
    #     if index % 10 == 0:
    #         sampleNumbers.append(index)
    #         orderedGruen.append(index/10)

    plt.figure().clear()
    plt.close('all')

    plt.plot(sampleNumbers, label="Amostras")
    plt.plot(datasetNumbers, label="Base de Dados")

    plt.legend()
    plt.title(title)
    outfile = outfile + '.pdf'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')


def calcStatistics(sampleScores, datasetScores):
    allScores = []
    grammScores = []
    focusScores = []
    redundancyScore = []
    for index, sampleScore in enumerate(sampleScores):
        allScores.append(float(sampleScore[1]))
        grammScores.append(float(sampleScore[GRAMATICALITY]))
        redundancyScore.append(float(sampleScore[REDUNCANCY]))
        focusScores.append(float(sampleScore[FOCUS]))

    allScoresDataset = []
    grammScoresDataset = []
    focusScoresDataset = []
    redundancyScoreDataset = []
    for index, sampleScore in enumerate(datasetScores):
        allScoresDataset.append(float(sampleScore[1]))
        grammScoresDataset.append(float(sampleScore[GRAMATICALITY]))
        redundancyScoreDataset.append(float(sampleScore[REDUNCANCY]))
        focusScoresDataset.append(float(sampleScore[FOCUS]))

    boxplotTwoInstances(grammScores, grammScoresDataset, "Grammaticality Score", "boxplotGruenGrammScoreComparison")
    boxplotTwoInstances(focusScores, focusScoresDataset, "Focus Score", "boxplotGruenFocusScoreComparison")
    boxplotTwoInstances(redundancyScore, redundancyScoreDataset, "Redundancy Score",
                        "boxplotGruenRedundancyScoreComparison")

    plotLines(allScores, allScoresDataset, "Scores", "allscoresline")
    plotLines(grammScores, grammScoresDataset, "Grammaticality", "grammline")
    plotLines(focusScores, focusScoresDataset, "Focus", "focusLine")
    plotLines(redundancyScore, redundancyScoreDataset, "Redundancy", "redLine")
    print("Característica Avaliada & Média base de dados & Média amostras \\\\")
    print("Pontuação geral &", "$", "{:.4f}".format(statistics.mean(allScoresDataset)), "$", "&", "$",
          "{:.4f}".format(statistics.mean(allScores)), "$", "\\\\")
    print("Pontuação de Gramaticalidade &", "$", "{:.4f}".format(statistics.mean(grammScoresDataset)), "$", "&", "$",
          "{:.4f}".format(statistics.mean(grammScores)), "$",
          "\\\\")
    print("Pontuação de Foco &", "$", "{:.4f}".format(statistics.mean(focusScoresDataset)), "$", "&", "$",
          "{:.4f}".format(statistics.mean(focusScores)), "$", "\\\\")
    print("Pontuação de  Redundância &", "$", "{:.4f}".format(statistics.mean(redundancyScoreDataset)), "$", "&", "$",
          "{:.4f}".format(statistics.mean(redundancyScore)), "$",
          "\\\\")
    print()
    print()

    print("Característica Avaliada & Variação padrão base de dados & Variação padrão amostras \\\\")
    print("Pontuação geral &", "$", "{:.4f}".format(statistics.stdev(allScoresDataset)), "$", "&", "$",
          "{:.4f}".format(statistics.stdev(allScores)), "$", "\\\\")
    print("Pontuação de Gramaticalidade &", "$", "{:.4f}".format(statistics.stdev(grammScoresDataset)), "$", "&", "$",
          "{:.4f}".format(statistics.stdev(grammScores)), "$",
          "\\\\")
    print("Pontuação de Foco &", "$", "{:.4f}".format(statistics.stdev(focusScoresDataset)), "$", "&", "$",
          "{:.4f}".format(statistics.stdev(focusScores)), "$", "\\\\")
    print("Pontuação de  Redundância &", "$", "{:.4f}".format(statistics.stdev(redundancyScoreDataset)), "$", "&", "$",
          "{:.4f}".format(statistics.stdev(redundancyScore)), "$",
          "\\\\")

    # print("average general scores samples", statistics.mean(allScores))
    # print("average grammatically scores samples", statistics.mean(grammScores))
    # print("average focus scores samples", statistics.mean(focusScores))
    # print("average redundancy scores samples", statistics.mean(redundancyScore))
    # print()
    # print()
    # print("average general scores dataset", statistics.mean(allScoresDataset))
    # print("average grammatically scores dataset", statistics.mean(grammScoresDataset))
    # print("average focus scores dataset", statistics.mean(focusScoresDataset))
    # print("average redundancy scores dataset", statistics.mean(redundancyScoreDataset))


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



    sampleTrainingFilteredScores = []

    sampleTrainingFilteredGrammaticality_score = []

    sampleTrainingFilteredRedundancy_score = []

    sampleTrainingFilteredFocus_score = []
    sampleTrainingFilteredNumber = []
    sampleTrainingFilteredAllScores = []
    with open("gruenScoreSamplesAllScores-onlyHighScoresDataset.csv", newline='', encoding="utf8") as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            try:
                gruen_score = float(row['gruen score'])
                grammaticality_score = float(row['grammaticality_score'])
                redundancy_score = float(row['redundancy_score'])
                focus_score = float(row['focus_score'])
                sampleTrainingFilteredScores.append(gruen_score)
                sampleTrainingFilteredGrammaticality_score.append(grammaticality_score)
                sampleTrainingFilteredRedundancy_score.append(redundancy_score)
                sampleTrainingFilteredFocus_score.append(focus_score)
                fileName = row['fileName']
                fileName = fileName.replace('samples-', '')
                sampleTrainingFilteredNumber.append(int(fileName))
                sampleTrainingFilteredAllScores.append(
                    [int(fileName), gruen_score, grammaticality_score, redundancy_score, focus_score])
            except:
                print('error')
                x = 0
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

    plotScore(sampleAllScores, "Pontuações das Amostras", "gruenSampleScore")
    plotScore(sampleAllScores, "Pontuações das Amostras", "gruenSampleScoreWithZero", True)
    plotScore(sampleTrainingFilteredAllScores, "Pontuações das Amostras (treinamento com base de dados filtrada)", "gruenSampleTrainingFilteredScore")
    plotScore(sampleTrainingFilteredAllScores, "Pontuações das Amostras (treinamento com base de dados filtrada)", "gruenSampleScoreTrainingFilteredWithZero", True)
    plotScore(datasetAllScores, "Pontuações dos textos da Base de Dados", "gruenDatasetScore")
    plotScore(datasetAllScores, "Pontuações dos textos da Base de Dados", "gruenDatasetScoreZero", True)

    # boxplotSampleResults(sampleAllScores, datasetAllScores, True)
    boxplotSampleResults(sampleAllScores, datasetAllScores)

    plotTrend(sampleAllScores, "Linha de tendência da pontuação geral das amostras", 'gruenSampleTrend')
    plotTrend(sampleAllScores, "Linha de tendência da pontuação geral das amostras", 'gruenSampleTrendWithZero', True)
    plotTrend(sampleTrainingFilteredAllScores, "Linha de tendência da pontuação geral das amostras", 'gruenSampleTrainingFilteredTrend')
    plotTrend(sampleTrainingFilteredAllScores, "Linha de tendência da pontuação geral das amostras", 'gruenSampleTrainingFilteredTrendWithZero', True)
    plotTrend(datasetAllScores, "Linha de tendência da pontuação geral dos textos da Base de Dados",
              'gruenDatasetTrend')

    sampleAllScores.sort(key=sortByScore)
    datasetAllScores.sort(key=sortByScore)

    plotScoreHistogramSample(sampleAllScores)
    plotScoreHistogramDataset(datasetAllScores)

    # region statistical calculations

    calcStatistics(sampleAllScores, datasetAllScores)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
