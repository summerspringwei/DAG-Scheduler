import argparse
import logging
import subprocess

def parse_model_mobile():
    model_list = [
        'inception-v3', 'inception-v4', 'lanenet', 'pnasnet-large',
        'pnasnet-mobile', 'nasnet-mobile', 'nasnet-large',
        'inception-resnet-v2', 'model1', 'model2', 'model3', 'model4',
        'example1'
    ]
    mobile_list = [
        'lenovo_k5', 'redmi', 'vivo_z3', 'oneplus5t', 'huawei_mate_20',
        'snapdragon_855', 'huawei_p40', 'device1', 'device2', 'device3'
    ]
    thread_number = [1, 2, 4, 8]

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('model', type=str, help='Enter the model name')
    parser.add_argument('mobile', type=str, help='Enter the mobile name')
    parser.add_argument('thread', type=int, help='Enter the thread number')
    parser.add_argument("num_little_thread", type=int, \
        help='Optional: Enter the number of little thread', nargs='?', default=None)
    args = parser.parse_args()
    model = args.model
    mobile = args.mobile
    thread = args.thread
    num_little_thread = args.num_little_thread
    print("Get model name %s mobile %s thread %d" % (model, mobile, thread))
    # Check args
    if model not in model_list:
        print("Model name %s not support yet. Exit now." % model)
        exit(0)
    if mobile not in mobile_list:
        print("Mobile name %s not support yet. Exit now." % mobile)
        exit(0)
    if thread not in thread_number:
        print("Thread number %d not avaliable. Exit now." % thread)
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