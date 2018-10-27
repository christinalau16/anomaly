import itertools

def loadData():
    HEADER_COUNT = 1490 + 1490 + 6
    FOOTER_COUNT = 1
    TOTAL_COUNT = 2223087

    with open('data/datasetPolitical/blogs.dat.txt') as infile, open('data/datasetPolitical/datasetPolitical_final.csv','w') as outfile:
        for line in itertools.islice(infile, HEADER_COUNT, TOTAL_COUNT - FOOTER_COUNT):
            outfile.write(line.strip().replace(' ',',') + "\n")

loadData()
