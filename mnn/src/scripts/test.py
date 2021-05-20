import os

model_name1 = ["inception-v3", "inception-v4", "pnasnet-mobile", "pnasnet-large", "nasnet-large"]
model_name2 = ["inception_v3", "inception_v4", "pnasnet_mobile", "pnasnet_large", "nasnet_large"]

mobile="vivo_z3"
for i in range(len(model_name1)):
    run_sh = "./scripts/bench_alone.sh {} {}".format(model_name2[i], mobile)
    print(run_sh)
    os.system(run_sh)
    # pull_sh = "adb pull /data/local/tmp/{} ../models/acl-{}/acl-{}-info.txt".format(model_name2[i], model_name2[i], model_name2[i])
    # print(pull_sh)
    # os.system(pull_sh)
    # rm_sh = "rm ../../models/acl-{}/{}/acl-{}-{}-data-trans.csv".format(model_name2[i], mobile, model_name2[i], mobile)
    # print(rm_sh)
    # os.system(rm_sh)
    # cp_sh = "cp ../../models/{}/{}/{}-{}-data-trans.csv ../../models/acl-{}/{}/acl-{}-{}-data-trans.csv".format(\
    #     model_name1[i], mobile, model_name1[i], mobile, model_name2[i], mobile, model_name2[i], mobile)
    # print(cp_sh)
    # os.system(cp_sh)
