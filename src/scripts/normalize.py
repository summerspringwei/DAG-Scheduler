

f = open("./scripts/data.txt")
lines = f.readlines()

outlines = []
float_com = []
for line in lines:
    com = line.split('\t')
    com = [float(c) for c in com]
    com = [c/com[1] for c in com]
    float_com.append(com)
    line = ""
    for c in com:
        line += str(c)+','
    print(line)


