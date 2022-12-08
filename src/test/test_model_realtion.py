from read_profile_data import *


def test_connection():
    op_name_list, name_op_dict, net_def = gather_model_profile(
            "pnasnet-large/pnasnet-large-info.bak",
            "./inception-v3/redmi_data_trans.txt",
            "pnasnet-large/oneplus3-pnasnet-large-latency-onwait.csv")


    first_op_name = op_name_list[0]
    op_queue = []
    op_queue.append(first_op_name)

    while len(op_queue) > 0:
        op_name = op_queue.pop()
        print(op_name)
        op = name_op_dict[op_name]
        op.executed = True
        for child_name in op.children:
            if not name_op_dict[child_name].executed and child_name not in op_queue:
                op_queue.append(child_name)


def diff_two_file(f1_path, f2_path):
    pass