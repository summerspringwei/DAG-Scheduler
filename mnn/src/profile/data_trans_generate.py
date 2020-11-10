import numpy as np
import os
from analyze import measure_interference
from utils import utils

def avg_data_trans(model, mobile):
    # Read cpu-2-gpu and gpu-2-cpu multi tensor transform 
    file_name = model+'-'+mobile+'-c2g-multi-data-trans.txt'
    multi_data_trans_file_path = os.path.join("../models", model, mobile, file_name)
    profile_dict_c2g = measure_interference.read_multi_runs_latency(multi_data_trans_file_path)
    file_name = model+'-'+mobile+'-g2c-multi-data-trans.txt'
    multi_data_trans_file_path = os.path.join("../models", model, mobile, file_name)
    profile_dict_g2c = measure_interference.read_multi_runs_latency(multi_data_trans_file_path)

    result_file_name = model+'-'+mobile+'-data-trans.csv'
    result_file_path = os.path.join("../models", model, mobile, result_file_name)
    f = open(result_file_path, 'w')
    lines = []
    
    for name, values in profile_dict_c2g.items():
        trans_latency_c2g = np.average(measure_interference.filter_list(values[2]))
        if name in profile_dict_g2c.keys():
            trans_latency_g2c = np.average(measure_interference.filter_list(profile_dict_g2c[name][2]))
        else:
            trans_latency_g2c = trans_latency_c2g
        line = "%s %f %f\n" % (name, trans_latency_c2g, trans_latency_g2c)
        lines.append(line)
    f.writelines(lines)
    f.flush()
    f.close()
    print('Write data trans to %s' % result_file_path)


if __name__ == "__main__":
    model, mobile, thread, _ = utils.parse_model_mobile()
    avg_data_trans(model, mobile)
    