def read_names(file_path):
    f = open(file_path, 'r')
    names = set()
    lines = f.readlines()
    for line in lines:
        names.add(line.strip())
    return names


def compare(f1, f2):
    model1 = read_names(f1)
    model2 = read_names(f2)
    for name in model1:
        if name not in model2:
            print(name)
        else:
            model2.remove(name)
    for name in model2:
        print(name)


if __name__ == "__main__":
    compare("pnasnet-large/pnasnet-large-info.bak", "tmp.txt")
