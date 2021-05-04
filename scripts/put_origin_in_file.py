infile = open("/home/ida/master-thesis/data/temporary_data/all_data_numbered.csv")
outfile = open("/home/ida/master-thesis/data/temporary_data/all_data_numbered_origin.csv", "w")

tenX = True
swapped = False
positive = False
for line in infile:
    if line.startswith("#"):
        outfile.write(line.strip()+",origin\n")
        continue
    if tenX:
        if "AVSSFSGGYNKLI" in line and "SASRQWAQQETQY" in line:
            swapped = True
            tenX = False
        else:
            outfile.write(line.strip() + ",tenX\n")
    if swapped:
        if line.strip().split(",")[5] == "1":
            swapped = False
            positive = True
        else:
            outfile.write(line.strip() + ",swapped\n")
    if positive:
        outfile.write(line.strip()+",positive\n")

infile.close()
outfile.close()
