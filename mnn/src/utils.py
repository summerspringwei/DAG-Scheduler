import argparse

def parse_model_mobile():
    model_list = ['inception-v3', 'inception-v4', 'lanenet', 'pnasnet-large']
    mobile_list = ['lenovo_k5', 'redmi', 'vivo_z3', 'oneplus5']
    thread_number = [1, 2, 4, 8]

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('model', type=str, help='Enter the model name')
    parser.add_argument('mobile', type=str, help='Enter the mobile name')
    parser.add_argument('thread', type=int, help='Enter the thread number')

    args = parser.parse_args()
    model = args.model
    mobile = args.mobile
    thread = args.thread
    # Check args
    if model not in model_list:
        print("Model name %s not support yet. Exit now." % model)
        exit(0)
    if mobile not in mobile_list:
        print("Mobile name %s not support yet. Exit now." % mobile)
        exit(0)
    if thread not in thread_number:
        print("Thread number %d not avaliable. Exit now." % thread)
    return model, mobile, thread
