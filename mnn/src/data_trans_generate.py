import numpy as np
from measure_inteference import *
from utils import *

def avg_data_trans(model, mobile):
    file_name = model+'-'+mobile+'-multi-data-trans.txt'
    multi_data_trans_file_path = os.path.join("../models", model, mobile, file_name)
    profile_dict = read_multi_runs_latency(multi_data_trans_file_path)
    file_name = model+'-'+mobile+'-multi-data-trans.txt'
    result_file_name = model+'-'+mobile+'-data-trans.csv'
    result_file_path = os.path.join("../models", model, mobile, result_file_name)
    f = open(result_file_path, 'w')
    lines = []
    for name, values in profile_dict.items():
        trans_latency = np.average(values[2])
        line = "%s %f %f\n" % (name, trans_latency, trans_latency)
        lines.append(line)
    f.writelines(lines)
    f.flush()
    f.close()
    print('Write data trans to %s' % result_file_path)


if __name__ == "__main__":
    model, mobile, thread = parse_model_mobile()
    avg_data_trans(model, mobile)
    