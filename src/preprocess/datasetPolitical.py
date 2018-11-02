import itertools

def loadData(delimiter):
    HEADER_COUNT = 1490 + 1490 + 6
    FOOTER_COUNT = 1
    TOTAL_COUNT = 2223087

    lookup = dict()

    with open('data/datasetPolitical/blogs.dat.txt') as infile, open('data/datasetPolitical/datasetPolitical_final.csv','w') as outfile:
        for line in itertools.islice(infile, HEADER_COUNT, TOTAL_COUNT - FOOTER_COUNT):
            src, dest, weight = line.strip().split(delimiter)
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
            outfile.write(str(src_id) + "," + str(dest_id) + "," + str(weight) + "\n")
loadData(' ')
