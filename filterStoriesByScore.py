import csv
import codecs
with open("datasetWithScore.csv", newline='', encoding="utf8") as csvfile:
     spamreader = csv.DictReader(csvfile)
     file = codecs.open("storiesWithHighScore", "w", "utf-8")
     for row in spamreader:
        # print(row['fileName'], )
        try:
            gruen_score = float(row['gruen score'])
            story = row['story']
            # print(gruen_score)
            tmp = '---------------------------------------------------'
            if (gruen_score > 0.4):
                file.write(story)
                file.write(tmp)
        except:
            x = 0
        #  if row[2] != 0:
        #      print(row[1])
     file.close()