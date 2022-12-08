import json

from profile import net_struct
from utils import utils


def read_contend(file_path):
    """Fetches all the content in the file specified by the file_path
    """

    f = open(file_path, 'r')
    try:
        content = f.read()
    finally:
        f.close()
    return content


def read_hw_json(json_file_path):
    """Parse mindspore lite json file

    """
    j_str = read_contend(json_file_path)
    j_obj = json.loads(j_str)
    internal_tensors = []
    internal_tensors.extend(j_obj["inputIndex"])
    nodes = j_obj["nodes"]
    for node in nodes:
        internal_tensors.extend(node["outputIndex"])
    op_list = []

    for node in nodes:
        name = str(node["name"])
        input_index = [
            idx for idx in node["inputIndex"] if idx in internal_tensors
        ]
        output_index = node["outputIndex"]
        op = net_struct.Operator(name)
        for idx in input_index:
            op.input_tensors.append((str(idx), name))
        for idx in output_index:
            op.output_tensors.append((str(idx), name))
        op_list.append(op)

    return op_list


def write_model_info(op_list, file_path):
    """Store the ops in op_list to my format and write the results to the file_path
    """
    lines = []
    for op in op_list:
        line = op.name + " "
        for addr, shape in op.input_tensors:
            line += (shape + "@" + addr + ";")
        line += " "
        for addr, shape in op.output_tensors:
            line += (shape + "@" + addr + ";")
        line += "\n"
        lines.append(line)
    utils.write_lines(file_path, lines)



def convert_model(model_name):
    """Test the functionality of this file
    """
    op_list = read_hw_json("../models/{}/{}.json".format(model_name, model_name))
    op_list = net_struct.build_op_relationship(op_list)
    write_model_info(op_list, "../models/{}/{}-info.txt".format(model_name, model_name))


if __name__ == "__main__":
    convert_model("model4")

