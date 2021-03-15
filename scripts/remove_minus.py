infile = open("/home/ida/master-thesis/data/all_data_final.csv")
outfile = open("/home/ida/master-thesis/data/all_data_final_no_minus.csv", "w")

for line in infile:
    outfile.write(line.replace("-", ""))

infile.close()
outfile.close()