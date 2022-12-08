
from solver import scheduler_utils

def depth_first_reorder(lines, op_name_list, name_op_dict):
    device_placement_dict = {}
    op_name_device_list = []
    for line in lines:
        com = line.split(" ")
        device_placement_dict[com[0].strip()] = com[1].strip()
        op_name_device_list.append(com[0])
    
    new_device_placement_lines = []
    visited = set()
    for op_name in op_name_device_list:
        if op_name in visited:
            continue
        device = device_placement_dict[op_name]
        assert(op_name in name_op_dict.keys())
        op = name_op_dict[op_name]
        new_device_placement_lines.append("%s %s\n" % (op_name, device))
        visited.add(op_name)
        op.executed = True
        for child_name in op.children:
            child = name_op_dict[child_name]
            child_device = device_placement_dict[child_name]
            if child_device == device and \
                scheduler_utils.is_parents_executed(child, op_name_list, name_op_dict):
                new_device_placement_lines.append("%s %s\n" % (child_name, device))
                visited.add(child_name)

    return new_device_placement_lines
