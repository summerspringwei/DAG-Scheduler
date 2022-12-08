
from operator import attrgetter

from utils import utils
from solver import scheduler_utils
from profile import read_profile_data

logger = utils.get_logger()

def binary_partitioning(ops_list, uprank_count_list, acc_uprank_list, start_idx, end_idx, level, NUM_OP_THRESHOLD, BALANCE_FACTOR):
    min_vertex_cut = 100000
    min_balance_gap = 100000
    cut_uprank = 0
    total_size = acc_uprank_list[end_idx] - acc_uprank_list[start_idx-1]
    # if len(ops_list) != total_size:
    #     logger.error("Graph num of ops is not equal, \
    #         len(ops_size) is {} while acc_uprank_list is{}".format(len(ops_list), total_size))
    
    v0_range, v1_range = [], []
    v0_size, v1_size = 0, 0
    can_find_balance = False
    
    # Find partition with the minimum vertex cut and also satisfy balance constraint
    tab_str = "\t"*level
    while not can_find_balance:
        balance_threshold = (1 + BALANCE_FACTOR) * (total_size / 2.0)
        for i in range(start_idx, end_idx):
            # i+1 as our uprank starts from 1 but i starts from 0
            v0_size = acc_uprank_list[i] - acc_uprank_list[start_idx-1]
            v1_size = acc_uprank_list[end_idx] - acc_uprank_list[i]
            if v0_size > balance_threshold or v1_size > balance_threshold:
                continue
            # Choose the partition with min_vertex_cut  
            if uprank_count_list[i] <= min_vertex_cut and abs(v0_size - v1_size) < min_balance_gap:
                can_find_balance = True
                min_balance_gap = abs(v0_size - v1_size)
                min_vertex_cut = uprank_count_list[i]
                cut_uprank = i
                print("{}[{},{}]: candidate cut_uprank: {}, v0_size: {}, v1_size: {}, min_vertex_cut: {}, min_balance_gap: {}".format
                    (tab_str, cut_uprank, start_idx, end_idx, v0_size, v1_size, min_vertex_cut, min_balance_gap))
        if not can_find_balance:
            logger.info("""Cannot find any balanced solutions.\n 
            total_size: {} total_rank:{} required max balance size: {}, uprank:{}""".format
            (total_size, (end_idx-start_idx), balance_threshold, uprank_count_list[start_idx+1:end_idx+1]))
            BALANCE_FACTOR += 0.1
    
    split_v0_size = acc_uprank_list[cut_uprank] - acc_uprank_list[start_idx-1]
    split_v1_size = acc_uprank_list[end_idx] - acc_uprank_list[cut_uprank]
    
    if split_v0_size <= NUM_OP_THRESHOLD:
        v0_range = [(start_idx, cut_uprank)]
        print("{}put:[{},{}]: v0_size: {}".format(tab_str, start_idx, cut_uprank, split_v0_size))
    elif split_v0_size > NUM_OP_THRESHOLD and start_idx < cut_uprank:
        print("{}recursive:[{},{}]: v0_size: {}".format(tab_str, start_idx, cut_uprank, split_v0_size))
        v0_range = binary_partitioning(ops_list, uprank_count_list, acc_uprank_list, start_idx, cut_uprank, level+1, NUM_OP_THRESHOLD, BALANCE_FACTOR)
    if split_v1_size <= NUM_OP_THRESHOLD:
        v1_range = [(cut_uprank+1, end_idx)]
        print("{}put:[{},{}]: v1_size: {}".format(tab_str, cut_uprank+1, end_idx, split_v1_size))
    elif split_v1_size > NUM_OP_THRESHOLD and cut_uprank+1 < end_idx:
        v1_range = binary_partitioning(ops_list, uprank_count_list, acc_uprank_list, cut_uprank+1, end_idx, level+1, NUM_OP_THRESHOLD, BALANCE_FACTOR)
        print("{}recursive:[{},{}]: v1_size: {}".format(tab_str, cut_uprank+1, end_idx, split_v1_size))

    return v0_range + v1_range


def uprank_partitioning(op_name_list, name_op_dict, NUM_OP_THRESHOLD = 12, BALANCE_FACTOR = 0.2):
    scheduler_utils.compute_bottom_level(op_name_list, name_op_dict, scheduler_utils.BottomLevelFuncType.RANK)
    ops_list = [name_op_dict[op_name] for op_name in op_name_list]
    ops_list = sorted(ops_list, key=attrgetter("bottom_level"))
    for op in ops_list:
        logger.info("{} {}".format(op.name, op.bottom_level))
    if len(op_name_list) < NUM_OP_THRESHOLD:
        return
    uprank_count_list = [0] * (len(op_name_list) + 2)
    acc_uprank_list = [0] * (len(op_name_list) + 2)
    max_uprank = 1
    for op in ops_list:
        uprank_count_list[op.bottom_level] += 1
        if op.bottom_level > max_uprank:
            max_uprank = op.bottom_level
    acc_uprank_list[0] = 0
    print("uprank_count_list: {}, len: {}".format(uprank_count_list, len(uprank_count_list)))
    for i in range(0, max_uprank):
        acc_uprank_list[i+1] = acc_uprank_list[i] + uprank_count_list[i+1]
    print("acc_uprank_list: {}, len: {}".format(acc_uprank_list, len(acc_uprank_list)))
    # logger.info(acc_uprank_list)
    for uprank_count in uprank_count_list:
        if uprank_count > NUM_OP_THRESHOLD:
            NUM_OP_THRESHOLD = uprank_count
    uprank_parts = binary_partitioning(ops_list, uprank_count_list, acc_uprank_list, 1, max_uprank, 1, NUM_OP_THRESHOLD, BALANCE_FACTOR)
    print("uprank_parts: {}, len: {}".format(uprank_parts, len(uprank_parts)))
    subgraphs_list = []
    # bottom_level starts from 1
    total_num_ops = 0
    for (start_uprank, end_uprank) in uprank_parts:
        op_one_part = list()
        for op in ops_list:
            if op.bottom_level >= start_uprank and op.bottom_level <= end_uprank:
                op_one_part.append(op.name)
        subgraphs_list.append(op_one_part)
        if start_uprank != 0:
            start_uprank = start_uprank-1
        count = acc_uprank_list[end_uprank] - acc_uprank_list[start_uprank]
        total_num_ops += count
        if  count != len(op_one_part):
            print("{} != {}".format(count, len(op_one_part)))
    print("total_num_ops: {} len(op_name_list): {}".format(total_num_ops, len(op_name_list)))
    # assert(total_num_ops == len(op_name_list))
    return subgraphs_list


def write_partitioning_result(op_name_list, name_op_dict, subgraphs_list):
    part_idx = 0
    for op_one_part in subgraphs_list:
        for op_name in op_one_part:
            logger.info("{} {}".format(op_name, part_idx))
        part_idx+=1


def test_uprank_partitioning():
    model, mobile, thread, _ = utils.parse_model_mobile()
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread)
    subgraph_list = uprank_partitioning(op_name_list, name_op_dict)
    print(subgraph_list)

if __name__=="__main__":
    test_uprank_partitioning()
    