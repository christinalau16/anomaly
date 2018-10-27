
def loadData():
    with open('data/datasetEmail/email-Eu-core.txt') as infile, open('data/datasetEmail/datasetEmail_final.csv','w') as outfile:
        for line in infile:
            outfile.write(line.strip().replace(' ',',') + ",1\n" )

loadData()
