
input: netdef, input_node_name

def greedy_device_placement:
    CPU_end_point = 0
    GPU_end_point = 0
    # Need to record the execute order
    op_execute_order = list()
    # Used to maintain the ops that all his parents has been executed
    # thus it can be assigned to devices
    ready_to_assign_list
    add input op into ready_to_assign_list
    # Tranverse the DAG in the manner like BFS
    while True:
        for child in op.child:
            if child.parent executed:
                add to ready_to_assign_list
        # Fetch the op that can be executed as early as possible
        sort ready_to_assign_list according to start_point
        fetch the first op
        op_execute_order.append(op)
        if op is only support by CPU:
            assign to CPU
            continue
        # Latency consists of two parts: computation latency and transpose latency
        CPU_latency = CPU_computation_latency + Transpose_from_GPU_to_CPU_latency
        GPU_latency = GPU_computation_latency + Transpose_from_CPU_to_GPU_latency
        # Greedy strategy: for each op, find the device that can finish this op as `earlier` as possible. 
        # Note: Note as `fast` as possible but as `earlier` as possible
        # op can be executed at the very first time, but CPU and GPU are busy
        if CPU_end_point >= op.earlist_start_point and GPU_end_point >= op.earlist_start_point:
            if CPU_end_point + CPU_latency < GPU_end_point + GPU_latency: # CPU can finish this op earlier(Greedy here)
                assign op to CPU
            else: # GPU is better
                assign op to GPU
        # CPU is idel but GPU is busy(or just finish work)
        elif (op.earlist_start_point >= CPU_end_point and op.earlist_start_point <= GPU_end_point):
            if op.earlist_start_point + CPU_latency < GPU_end_point + GPU_latency:# Note, CPU_end_point changed
                CPU_end_point = op.earlist_start_point
                CPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.CPU, CPU_end_point, CPU_latency)
            else:
                GPU_end_point = assign_op_to_device(op, ops_relation_dict, DeviceType.GPU, GPU_end_point, GPU_latency)
        # Same for GPU
        update_children_start_point(op)
    generate new netdef based on op_execute_order
    return new netdef
