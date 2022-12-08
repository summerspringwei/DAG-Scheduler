# DAG-Scheduler
DAG-scheduler is an offline Directed-Acylic-Graph (DAG) schedule,
which can map operators of DNN models to multiple heterogenous computing units for running in parallel.
DAG-scheduler is a part of the HOPE project (Heterogeneous Oriented Parallel execution Engine)
and is responsible for generating execution plans.
The other part of HOPE is [parallelelmnn](https://github.com/summerspringwei/parallelmnn-patch.git),
which is responsible for executing the execution plans on mobile devices.

## prerequisite
Python libraries:
```shell
numpy
matplotlib
pysnooper
```
Other software:
1. DAG-scheduler relies on Integer Linear Programming Solver,
We need to install [GLPK](https://www.gnu.org/software/glpk/)(GNU Linear Programming Kit) and 
add the `glpsol` to the system path.
IBM CPLEX [CPLEX](https://www.ibm.com/support/pages/downloading-ibm-ilog-cplex-optimization-studio-v1290) is also prefered,
Add the path to `cplex` to the system path.
2. We also need [graphviz](https://graphviz.org/download/) to generate the execution plan to figures.
3. Android SDK platform-tools.

System requirments: `Linux` or `MacOS`.


## File structure

```
.
├── models (Model structure, the execution latency on mobile devices and execution plan.)
│  ├── inception-resnet-v2
│  ├── inception-v3
│  ├── inception-v4
│  ├── lanenet
│  ├── model1
│  ├── model2
│  ├── model3
│  ├── model4
| ├── ....
└── src
 ├── analyze (Compare and analyze the performance of serial execution and parallel execution)
 ├── out
 ├── parser (Convert Json format model to our custom format)
 ├── profile (Benchmark DNN model and generate the execution latency)
 ├── result_data
 ├── scripts (Some scripts for profiling and benchmarking)
 ├── solver (Core library of this repo for generating execution plan)
 ├── test
 ├── utils 
 └── visualization 
```

## How to connect to Android device
We can use add to connect the android device attached on a remote server by specifying the host and port.
```shell
adb -H host_ip -P port shell
```
For example:
```
adb -H 10.10.245.207 -P 7035 shell
```
Then we can execute commond on the android device.


## Data structure
### Variables
We set the root directory of DAG-Scheduler as `PROJECT_HOME`,
DNN model's name as `model_name`, mobile device as `mobile_name`,
and the number of threads as `thread_number`.
In particular `model_name` can be one the element in `{inception-v3, inception-v4, pnasnet-large, pnasnet-mobile, nasnet-large, nasnet-mobile}`.


### How to add new DNN models and mobile devices
For now, we support 8 public DNN models and 7 mobile devices.
DAG-Scheduler relies on the `model_name` and `mobile_name` to find corresponding data (e.g. operators' execution latency of `model_name` on `mobile_name`) in folder `models`.
If users want to add new DNN models and mobile devices,
follow the instructions listed below;

1. Create folders under `PROJECT_HOME/models`
```shell
mkdir -p $PROJECT_HOME/$model_name/$mobile_name
```

2. Add `model_name` and `mobile_name` in the local variables `model_list` and `mobile_list`
 of function `parse\_model\_mobile()`(in file `$PROJECT\_HOME/src/utils/utils.py`).

3. The DNN models graph topology is saved in `$PROJECT_HOME/models/$model_name/${model_name}-info.txt`.
Each row records the operator's name, input tensor's shape and id, output tensor's shape and id,
precedent operators and successive operators with the following format:
`op_name input_tensor_shape@input_tensor_id; output_tensor_shape@output_tensor_id; precedent_ops; sucessor_ops`

4. The execution latency for each operator on the mobile device is saved in `PROJECT_HOME/models/${model_name}/${mobile_name}/${mobile_name}-${model_name}-layerwise-latency.csv`.
Each row records the operator's name, the execution latency using one, two and four CPU cores and GPU with the following format:
`op_name CPU_1_thread_latency CPU_2_threads_latency CPU_4_threads_latency GPU_latency`.
DAG-Scheduler uses this file to get the profiling data.

5. The communication latency between CPU and GPU is saved in `$PROJECT_HOME/models/${model_name}/${mobile_name}/${model_name}-${mobile_name}-data-trans.csv`.
Each row records the tensor's shape, communication latency from CPU to GPU and communication latency from GPU to CPU with the following format:
`tensor_shape CPU_to_GPU_communication_latency GPU_to_CPU_communication_latency`.

6. With all the data mentioned before, DAG-Scheduler can be used to generate the execution plan.
Users need to convert their data to the corresponding formats before using DAG-Scheduler.

### MNN models
The converted MNN models can be downloaded from [onedrive](https://1drv.ms/u/s!AiO8PwT1yve8hLESiPBwTycYnjbFqg?e=TMvJ0Q).
Then push models to android devices:
```shell
adb push mnn_models /data/local/tmp
```

For pnasnet, there are some operators that is not supported by mobile GPU and are scheduled to CPU, we need to push the execution plan to Android device.
```shell
adb push $PROJECT_HOME/models/pnasnet-large/pnasnet-large-final-layer.txt /data/local/tmp/
```

## Profile execution latency
1. The execution engine is based on [MNN](https://github.com/alibaba/MNN.git).
Follow [this](https://github.com/summerspringwei/parallelmnn-patch.git) to build the runtime.

2. benchmark:
```shell
cd $PROJECT_HOME/mnn/src
./scripts/bench_alone.sh $model_name $mobile_name $adb_config
```
e.g.
```shell
./scripts/bench_alone.sh pnasnet-large huawei_p40 "-H 10.108.245.207 -P 7035"
```
The script will automatically benchmark `model_name` on the CPU (with one, two and four theads) and GPU,
the operators' execution latency would be saved to `$PROJECT_HOME/models/${model_name}/${mobile_name}/${mobile_name}-${model_name}-layerwise-latency.csv`

## Schedule DNN models
For now we support two scheduler: an ILP-based scheduler and a greedy strategy based schedule.
For ILP-based scheduler, the number of operators for DNN model is limited as the solving time would greate increase with the number of operators.
Thus we proposed a divide-and-conquer stratege to partition the DNN graph to multi-subgraphs,
 and solve each subgraph indvidually .
For now, the divide-and-conquer stratege only supports inception and nasnet models.

Run ILP-based scheduler:
```shell
python3 solver/ilp_device_placement.py model_name mobile_name thread 
```
e.g.
```shell
python3 solver/ilp_device_placement.py dfmodel1 npu 1
```

Run greedy-strategy-based scheduler:
```shell
python3 solver/greedy_device_placement.py dfmodel1 npu 1
```
The execution plan would be saved to `models/model_name/mobile_name/`
and pushed to folder `/data/local/tmp/` of Android device.

Push the helper scripts to mobile:
```
adb push scripts/set_env.sh /data/local/tmp/
adb push scripts/grun_mnn.sh /data/local/tmp/
adb push scripts/run_mnn.sh /data/local/tmp/
```

Run the benchmark
```shell
cd /data/local/tmp
source set_env.sh
./run_mnn.sh model_name $thread_number
./grun_mnn.sh model_name $thread_number
```
Script `run_mnn.sh` profiles the execution plan generated with ILP-based scheduler,
while `grun_mnn.sh` profiles the execution plan generated with greedy-strategy-based scheduler.

## Visualize the execution plan
DAG-Schedule can visualize the execution plan by utilizing the graphviz.
Run 
```shell
python visualization/draw_dag.py $model_name $mobile_name $thread
```
e.g.
```shell
python3 visualization/draw_dag.py inception-v3 redmi 1
```

The figure would be saved in `PROJECT_HOME/models/model_name/mobile_name`。


## Contact Us

For any questions, feel free to send an e-mail to `xiachunwei@ict.ac.cn`

