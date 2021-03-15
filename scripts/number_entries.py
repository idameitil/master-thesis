infile = open("/home/ida/master-thesis/data/all_data.csv")
outfile = open("/home/ida/master-thesis/data/all_data_numbered.csv", "w")

ID = 1
for line in infile:
    if line.startswith("#"):
        outfile.write("#ID," + line[1:])
    else:
        outfile.write(str(ID) + "," + line)
        ID += 1