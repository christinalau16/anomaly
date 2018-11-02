#Converts graph dataset .txt files to .csv files
#Each line contains three comma-separated values representing an edge:
#<source-id>, <destination-id>, <edge-weight>
#If edge-weight is not available in graph data, leave as 1

#load data for no header, footer
def loadDataNoExtras(delimiter, boolWeight):
    lookup = dict()

    with open('data/datasetEmail/email-Eu-core.txt') as infile, open('data/datasetEmail/datasetEmail_final.csv','w') as outfile:
        for line in infile:
            #get source, destination, weight from line
            if boolWeight:
                src, dest, weight = line.strip().split(delimiter)
            else:
                src, dest = line.strip().split(delimiter)

            #check source node
            if src in lookup:
                src_id = lookup[src]
            else:
                lookup[src] = len(lookup)
                src_id = len(lookup) - 1

            #check destination node
            if dest in lookup:
                dest_id = lookup[dest]
            else:
                lookup[dest] = len(lookup)
                dest_id = len(lookup) - 1

            #write new line to output file
            if boolWeight:
                outfile.write(str(src_id) + "," + str(dest_id) + "," + str(weight) + "\n")
            else:
                outfile.write(str(src_id) + "," + str(dest_id) + ",1\n" )
