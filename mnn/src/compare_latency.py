from measure_inteference import *

if __name__ == "__main__":
    model, mobile, thread = parse_model_mobile()
    model_dir = os.path.join("../models/", model)
    file_prefix = mobile+"-"+model
    result_file_path = os.path.join(model_dir, mobile, file_prefix+"-parallel-layerwise-compare.csv")
    
    gather_net_profile(os.path.join(model_dir, mobile, file_prefix+"-cpu-"+ str(thread) +".csv"), \
        os.path.join(model_dir, mobile, file_prefix+"-gpu-1.csv"), \
        os.path.join(model_dir, mobile, file_prefix+"-parallel-cpu-"+ str(thread) +".csv"), \
        os.path.join(model_dir, model+"-info.txt"), \
        result_file_path)
    
    print("Compare profile data for model %s on mobile %s done, write result to %s" % (model, mobile, result_file_path))
    