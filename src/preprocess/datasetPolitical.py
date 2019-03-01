import itertools
import pickle

def loadData(delimiter):
    HEADER_COUNT = 1490 + 1490 + 6
    FOOTER_COUNT = 1
    TOTAL_COUNT = 2223087

    lookup = dict()
    lookup_mapping = dict() #old node number to new node number
    lookup_classify = dict() #new node number to class number

    with open('data/datasetPolitical/blogs.dat.txt') as infile, open('data/datasetPolitical/datasetPolitical_final.csv','w') as outfile:
        for line in itertools.islice(infile, HEADER_COUNT, TOTAL_COUNT - FOOTER_COUNT):
            src, dest, weight = line.strip().split(delimiter)
            if weight != "0":
                if src in lookup:
                    src_id = lookup[src]
                else:
                    #lookup[src] = src
                    lookup[src] = len(lookup)
                    #src_id = src
                    src_id = len(lookup) - 1
                    lookup_mapping[src] = src_id
                    if (int(src) < 758):
                        lookup_classify[src_id] = 0
                    else:
                        lookup_classify[src_id] = 1

                if dest in lookup:
                    dest_id = lookup[dest]
                else:
                    #lookup[dest] = dest
                    lookup[dest] = len(lookup)
                    #dest_id = dest
                    dest_id = len(lookup) - 1
                    lookup_mapping[dest] = dest_id
                    if (int(dest) < 758):
                        lookup_classify[dest_id] = 0
                    else:
                        lookup_classify[dest_id] = 1

                outfile.write(str(src_id) + "," + str(dest_id) + "," + str(weight) + "\n")

    with open('data/datasetPolitical/blogs_mapping.pkl', 'wb') as pickle_file:
        pickle.dump(lookup_mapping, pickle_file)
    with open('data/datasetPolitical/blogs_classify.pkl', 'wb') as pickle_file:
        pickle.dump(lookup_classify, pickle_file)

loadData(' ')
b= pickle.load(open('data/datasetPolitical/blogs_mapping.pkl', 'rb'))
print(b)
c= pickle.load(open('data/datasetPolitical/blogs_classify.pkl', 'rb'))
print(c)
