import csv
with open("datasetWithScore.csv", newline='', encoding="utf8") as csvfile:
     spamreader = csv.DictReader(csvfile)
     for row in spamreader:
        # print(row['fileName'], )
        try:
            gruen_score = float(row['gruen score'])
            story = row['story']
            # print(gruen_score)
            if (gruen_score > 0.4):
                print('---------------------------------------------------')
                print(story)
        except:
            x = 0
        #  if row[2] != 0:
        #      print(row[1])