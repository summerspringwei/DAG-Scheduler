import json
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


def get_model_info(json_file_path, info_file_path):
    content = read_contend(json_file_path)
    j_obj = json.loads(content)
    ops = j_obj["graph"][0]["op"]
    lines = []
    tensor_shape = "1,1,1,1"
    for op in ops:
        src_name = None
        des_name = None
        line = None
        src_tensor_str = ""
        dst_tensor_str = ""
        op_name = op["name"]
        if "src_name" in op:
            src_name = op["src_name"]
            for src in src_name:
                src_tensor_str += "{}@{};".format(tensor_shape, src+"-tensor")
        else:
            src_tensor_str = "{}@{};".format(tensor_shape, op_name+"-tensor")
        if "dst_name" in op:
            dst_name = op["dst_name"]
            for dst in dst_name:
                dst_tensor_str += "{}@{};".format(tensor_shape, op_name+"-tensor")
        else:
            dst_tensor_str = "{}@{};".format(tensor_shape, op_name+"-tensor")
        
        line = "{} {} {}\n".format(op_name, src_tensor_str, dst_tensor_str)
        
        lines.append(line)
    utils.write_lines(info_file_path, lines)


if __name__=="__main__":
    # model, mobile, thread, _ = utils.parse_model_mobile()
    json_file_path = "../models/dfmodel2/inceptionv3_resize_bs1.json"
    info_file_path = "../models/{}/{}-info.txt".format("dfmodel2", "dfmodel2")
    get_model_info(json_file_path, info_file_path)
