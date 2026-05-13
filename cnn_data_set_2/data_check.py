with open("data_set/data_set_2.csv", "r") as f:
    rows = [line.strip().split("\t") for line in f]

#print(rows)

for i in range(0, len(rows), 2):
    if rows[i][0] != rows[i+1][0] or rows[i][1] != rows[i+1][1]:
        print(i)
        print(rows[i])
        print(rows[i+1])



