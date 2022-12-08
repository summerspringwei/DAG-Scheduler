import argparse
import logging
import subprocess

class ArgConfig:
    DEFAULT = 0
    SOLVER_GREEDY = 1
    SOLVER_ILP = 2
    

def parse_model_mobile(arg_config=ArgConfig.DEFAULT):
    model_list = [
        'inception-v3', 'inception-v4', 'lanenet', 'pnasnet-large',
        'pnasnet-mobile', 'nasnet-mobile', 'nasnet-large',
        'inception-resnet-v2', 'model1', 'model2', 'model3', 'model4',
        'example1', 'dfmodel1', 'dfmodel2', 'inceptionpart',
        'acl-alexnet', 'acl-alexnet_ulayer', 'acl-inception_v3',
        'acl-inception_v4', 'acl-nasnet_large', 'acl-nasnet_mobile',
        'acl-pnasnet_large', 'acl-pnasnet_mobile', 'acl-squeezenet',
        'acl-mobilenet', 'acl-mobilenet_v2'
    ]
    mobile_list = [
        'lenovo_k5', 'redmi', 'vivo_z3', 'oneplus5t', 'huawei_mate_20',
        'snapdragon_855', 'huawei_p40', 'device1', 'device2', 'device3', 'mi9',
        "npu"
    ]
    thread_number = [1, 2, 4, 8]
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('model', type=str, help='Enter the model name')
    parser.add_argument('mobile', type=str, help='Enter the mobile name')
    parser.add_argument('thread', type=int, help='Enter the thread number')
    parser.add_argument("num_little_thread", type=int, \
        help='Optional: Enter the number of little thread', nargs='?', default=None)
    if arg_config == ArgConfig.SOLVER_GREEDY:
        parser.add_argument("--search_window", type=int, \
            help='Optional: search', nargs='?', default=3)
    elif arg_config == ArgConfig.SOLVER_ILP:
        parser.add_argument("--uprank_size", type=int, \
            help='Optional: uprank', nargs='?', default=10)
    args = parser.parse_args()
    model = args.model
    mobile = args.mobile
    thread = args.thread
    num_little_thread = args.num_little_thread
    print("Get model name %s mobile %s thread %d" % (model, mobile, thread))
    # Check args
    if str(model) not in model_list:
        print("Model name %s not support yet. Exit now." % model)
        exit(0)
    if mobile not in mobile_list:
        print("Mobile name %s not support yet. Exit now." % mobile)
        exit(0)
    if thread not in thread_number:
        print("Thread number %d not avaliable. Exit now." % thread)
    if arg_config == ArgConfig.SOLVER_GREEDY:
        return model, mobile, thread, num_little_thread, args.search_window
    elif arg_config == ArgConfig.SOLVER_ILP:
        return model, mobile, thread, num_little_thread, args.uprank_size
    else:
        return model, mobile, thread, num_little_thread


def write_lines(file_path, lines):
    f = open(file_path, 'w')
    try:
        f.writelines(lines)
        f.flush()
    finally:
        f.close()


def get_logger():
    """Returns the logger to replace the print function
    """
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    logger = logging.getLogger()
    return logger

def get_project_path():
    sh_cmd = "cd ../../ && pwd"
    result = str(subprocess.check_output(sh_cmd, shell=True), encoding = "utf-8").strip()
    return result