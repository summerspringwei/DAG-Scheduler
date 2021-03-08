
from operator import attrgetter

from utils import utils
from solver import scheduler_utils
from profile import read_profile_data

logger = utils.get_logger()

NUM_OP_THRESHOLD = 15
BALANCE_FACTOR = 0.5

def binary_partitioning(ops_list, uprank_count_list, acc_uprank_list,  start_idx, end_idx):
    min_vertex_cut = 100000
    cut_uprank = 0
    total_size = acc_uprank_list[end_idx] - acc_uprank_list[start_idx]
    # if len(ops_list) != total_size:
    #     logger.error("Graph num of ops is not equal, \
    #         len(ops_size) is {} while acc_uprank_list is{}".format(len(ops_list), total_size))
    balance_threshold = (1 + BALANCE_FACTOR) * (total_size / 2.0)
    v0_range, v1_range = [], []
    for i in range(start_idx, end_idx):
        # i+1 as our uprank starts from 1 but i starts from 0
        v0_size = acc_uprank_list[i+1] - acc_uprank_list[start_idx]
        v1_size = acc_uprank_list[end_idx] - acc_uprank_list[i+1]
        if v0_size > balance_threshold or v1_size > balance_threshold:
            continue
        if uprank_count_list[i+1] < min_vertex_cut:
            min_vertex_cut = uprank_count_list[i+1]
            cut_uprank = i+1
            if v0_size <= NUM_OP_THRESHOLD:
                v0_range = [(start_idx, cut_uprank)]
            else:
                v0_range = binary_partitioning(ops_list, uprank_count_list, acc_uprank_list, start_idx, cut_uprank)
            if v1_size <= NUM_OP_THRESHOLD:
                v1_range = [(cut_uprank+1, end_idx)]
            else:
                v1_range = binary_partitioning(ops_list, uprank_count_list, acc_uprank_list, cut_uprank+1, end_idx)
    return v0_range + v1_range


def uprank_partitioning(op_name_list, name_op_dict):
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
    for i in range(0, max_uprank):
        acc_uprank_list[i+1] = acc_uprank_list[i] + uprank_count_list[i+1]
    logger.info(acc_uprank_list)
    uprank_parts = binary_partitioning(ops_list, uprank_count_list, acc_uprank_list, 0, max_uprank)
    print(uprank_parts)
    subgraphs_list = []
    for (start_uprank, end_uprank) in uprank_parts:
        op_one_part = list()
        for op in ops_list:
            if op.bottom_level >=  start_uprank and op.bottom_level <= end_uprank:
                op_one_part.append(op.name)
        subgraphs_list.append(op_one_part)
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
    