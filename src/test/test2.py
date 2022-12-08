from operator import itemgetter, attrgetter

def compute_data_trans_intersection(cpu_data, gpu_data, convert_data):
    def add_latency(data):
        return [(start, start+latency) for (start, latency) in data]
    cpu_data = add_latency(cpu_data)
    gpu_data = add_latency(gpu_data)
    convert_data = add_latency(convert_data)
    print(cpu_data)
    print(gpu_data)
    print(convert_data)
    sum_of_intersection = 0
    for (cs, ce) in convert_data:
        for (gs, ge) in gpu_data:
            if cs >= gs and cs <= ge:
                sum_of_intersection += (min(ce, ge) - cs)
            elif gs > cs and gs < ce:
                sum_of_intersection += (min(ce, ge) - gs)
    cpu_max = max([endtime for (_, endtime) in cpu_data])
    gpu_max = max([endtime for (_, endtime) in gpu_data])
    convert_max = max([endtime for (_, endtime) in convert_data])
    endpoint = max([cpu_max, gpu_max, convert_max])

    return endpoint, sum_of_intersection

def main():
    cpu_data = [(0, 3), (3, 2), (6, 1)]
    gpu_data = [(0, 3), (3, 2), (6, 1), (8,2)]
    convert_data = [(3, 1), (5,1), (6,2), (10, 1)]
    endpoint, sum_of_intersection = compute_data_trans_intersection(cpu_data, gpu_data, convert_data)
    print(endpoint, sum_of_intersection)


def test_list_replace():
    a = ['a 3', 'b 3', 'c 0']
    for line in a:
        line = line.replace(" 3", ' 2')
        print(line)
    print(a)

class Stu:
    def __init__(self, age, name):
        super().__init__()
        self.age = age
        self.name = name
    def __str__(self):
        return "{}, {}".format(self.age, self.name)

def test_list_sorted():
    def stu_cmp(s1, s2):
        if s1.age<s2.age:
            return True
        else:
            return s1.name < s2.name
    s1 = Stu(10, 'b')
    s2 = Stu(10, 'a')
    
    l = [s1, s2]
    print(l[0],l[1])
    l = sorted(l, key=attrgetter('age', 'name'))
    print(l[0],l[1])


if __name__ == "__main__":
    # main()
    # test_list_replace()
    # test_list_sorted()
    count = 1
    for i in [2,3,4,5,6,7,8,9,10,11]:
        count = count * i
    print(count*2048)