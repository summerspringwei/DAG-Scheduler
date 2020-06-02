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


if __name__ == "__main__":
    main()