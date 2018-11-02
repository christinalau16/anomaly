
def loadData(delimiter):
    lookup = dict()

    with open('data/datasetEmail/email-Eu-core.txt') as infile, open('data/datasetEmail/datasetEmail_final.csv','w') as outfile:
        for line in infile:
            src, dest = line.strip().split(delimiter)
            if src in lookup:
                src_id = lookup[src]
            else:
                lookup[src] = len(lookup)
                src_id = len(lookup) - 1

            if dest in lookup:
                dest_id = lookup[dest]
            else:
                lookup[dest] = len(lookup)
                dest_id = len(lookup) - 1
            outfile.write(str(src_id) + "," + str(dest_id) + ",1\n" )

loadData(' ')
