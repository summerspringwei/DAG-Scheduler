

f_order = open("inception-v3-info.txt", 'r')
name_list = []
for line in f_order.readlines():
    name_list.append(line.strip().split(" ")[0].strip())
f_order.close()

f_profile = open("redmi-inteference.txt", 'r')
profile_dict = {}
for line in f_profile.readlines():
    com = line.strip().split(' ')
    profile_dict[com[0].strip()] = (com[0],com[1],com[2])
f_profile.close()

f_result = open("mDevicemap-inception-cpu-inteference-reordered.txt", 'w')
result_lines = []
for name in name_list:
    name, device, latency = profile_dict[name]
    line = ("%s %s %s\n" % (name, device, latency))
    result_lines.append(line)
f_result.writelines(result_lines)
f_result.flush()
f_result.close()