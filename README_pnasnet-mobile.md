# Pnasnet-mobile Linear programing 求解流程

# Step 1. 生成线性规划表达式求解
1. 在`generate_LP.py`中，`generateLP()`函数中，确保`gather_model_profile`函数中填入的分别为DNN模型的基本信息，不同大小的tensor CPU和GPU上转换的开销，和pnasnet-mobile每个op在CPU上1、2、4线程和GPU上的延迟三个路径的path; `associate_op_name_with_idx`为进行LP求解的subgraph的名称列表。
2. 最后的main中, `lp_file_path` `result_file_path` 分别是生成的LP的文件和调用glpsol求解产生的结果的文件
3. Execute python command `python generate_LP.py` 生成最终结果。

# Step 2. 将各个subgraph的分配结果生成具体op的分配结果
1. 修改`read_net_structure.py`中`write_device_placement_result`的参数，分别为在CPU上的subgraph和在GPU上的subgraph，以及最后的device placement的文件，之后`adb push `到手机上。

# Step 3. 在手机上执行
1. 执行 `benchmark.out pnasnet-mobile/ 10 3 1 2 1 mDevice_map_pnasnet-mobile-cell_0.txt > tmp.txt`, 产生结果日志文件。
2. `cat tmp.txt | grep Iter | awk '{print $3, $5, $6, $7, $8'} | grep pattern` 抓取需要的op
# Step 4. 生成Chrome Tracing文件
1. 修改`generate_tracing.py`中的结果文件路径，产生新的文件名末尾增加了`.json`。
2. Chrome浏览器中输入 `chrome://tracing/` ，然后load。
