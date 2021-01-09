# DAG-Scheduler
DAG-scheduler是一个静态调度器，将DNN模型的算子映射到异构的计算单元上并行执行。
DAG-Scheduler是HOPE（heterogeneous oriented parallel execution engine）的调度器部分，
负责读取profile的性能数据，生成执行计划。
[parallelelmnn](https://gitee.com/ict_huawei_2020/parallelmnn.git)是HOPE的运行时部分，
读取DAG-Scheduler的执行计划，异构并行的执行DNN模型。

## 安装要求
需要安装python3及以下python包：
```
numpy
matplotlib
pysnooper
```
使用线性规划求解器，需要安装[GLPK](https://www.gnu.org/software/glpk/)（GNU Linear Programming Kit），并将`glpsol`加到系统路径中。

可视化需要安装[graphviz](https://graphviz.org/download/)

需要安卓SDK platform-tools。主要是需要adb工具。

以上均需要将相应二进制文件路径加到系统路径中。

## 文件组织结构
`mace`文件夹已弃用。
`mnn`文件夹中为主要文件目录（忽略了一些重复或者不重要的文件夹）。
```
.
├── models （模型的结构以及在各个mobile上的性能数据、求解结果）
│   ├── inception-resnet-v2
│   ├── inception-v3
│   ├── inception-v4
│   ├── lanenet
│   ├── model1
│   ├── model2
│   ├── model3
│   ├── model4
|   ├── ....
└── src
    ├── __pycache__
    ├── analyze （比较分析并行运行和串行执行的算子性能数据）
    ├── out
    ├── parser  （将HW的json格式的模型转为自定义格式）
    ├── profile （benchmark DNN模型并生成profile的数据）
    ├── result_data
    ├── scripts （一些进行profile和benchmark的脚本）
    ├── solver  （进行调度的求解器核心库）
    ├── test
    ├── utils   （公共函数库）
    └── visualization （调度结果可视化的库）
```

## 说明
假设工程根目录HOPE-Scheduler为$PROJECT\_HOME。我们使用`$model_name`代表模型的名称，`$mobile_name`代表设备的名称，```$thread_number```代表线程的数量。
其中`$model_name`可以是`[inception-v3, inception-v4, pnasnet-large, pnasnet-mobile, nasnet-large, nasnet-mobile]`之一。
`$adb_config`是执行adb的配置，例如通过adb 访问连接在远程服务器上的设备时，需要加上host和port。
例如
```
adb -H 10.108.245.207 -P 7035 shell
```
就可以通过7035端口访问在10.108.245.207上的安卓手机了。



## 数据结构
目前已经包括了`inception-v3`、`inception-v4`、`pnasnet-large`、`pnasnet-mobile`等8个公共模型和4个华为私有模型，在`mnn/models`文件夹下可以看到所有支持的model。
已经支持了`redmi`等7个mobile平台。
由于框架通过模型和mobile的名称自动的到`mnn/models`文件夹下寻找对应的数据，如果需要添加新的模型和mobile平台（下同），则需要

1. 在models下面创建文件夹，，模型名称为 $model_name，mobile名称为$mobile_name：
```shell
mkdir $PROJECT_HOME/$model_name
mkdir $PROJECT_HOME/$model_name/$mobile_name
```
2. 在$PROJECT\_HOME/src/utils/utils.py中的`parse\_model\_mobile()`函数中的`model_list`和`mobile_list`中分别添加$model_name和$mobile_name。

3. 模型的拓扑结构保存在`$PROJECT_HOME/models/$model_name/${model_name}-info.txt`中。每一行分别代表
`op_name input_tensor_shape@input_tensor_id;(repeat) output_tensor_shape@output_tensor_id; precedent_ops; sucessor_ops`
通过读取模型文件就能够得到网络的拓扑结构。

4. 每个mobile的每个算子的时间保存在`$PROJECT_HOME/models/${model_name}/${mobile_name}/${mobile_name}-${model_name}-layerwise-latency.csv`中。每一行的数据分别代表
`op_name CPU_1_thread_latency CPU_2_threads_latency CPU_4_threads_latency GPU_latency`。通过这个文件读取profile的算子数据。

5. 每个tensor在CPU和GPU之间进行传输和数据格式转换的开销保存在`$PROJECT_HOME/models/${model_name}/${mobile_name}/${model_name}-${mobile_name}-data-trans.csv`中，其中每一行的数据代表`tensor_shape CPU_to_GPU_communication_latency GPU_to_CPU_communication_latency`。

6. 通过3、4、5中的文件，就可以得到要进行调度的所有数据了。用户可以先将自己的数据转换到上述的格式，再直接调用HOPE-Scheduler的API。

## MNN模型
MNN的模型可以从[onedrive](https://1drv.ms/u/s!AiO8PwT1yve8gotwXzCTumTw325OyQ?e=SxsmeE)下载。下载完成后，需要执行
```shell
adb push mnn_models /data/local/tmp

```
将模型push到安卓手机上。

pnasnet有几个算子GPU不支持,需要把不支持的算子放到CPU上执行，因此需要把pnasnet的算子映射文件push到手机上。
```shell
adb push $PROJECT_HOME/models/pnasnet-large/pnasnet-large-final-layer.txt /data/local/tmp/
```

## Host执行命令说明。
在以下如无特殊说明，均为在host上的$PROJECT\_HOME/mnn/src目录下执行命令。

## 获取profile数据。
1. 框架的运行时系统基于[Alibaba MNN](https://github.com/alibaba/MNN.git)。我们对运行时进行了一系列的修改，使得其能够使用CPU+GPU异构并行的执行DNN模型的推理。我们的源码仓库在[parallelmnn](https://gitee.com/ict_huawei_2020/parallelmnn.git)。
   1. 有两种方式可以得到运行时：
   2. 直接获取已经构建好的二进制文件(推荐)。在[mnn-model-zoo](https://gitee.com/ict_huawei_2020/mnn-model-zoo.git)中的parallelmnn-binary.zip，解压并将其中的二进制可执行文件push到手机上。
   ```shell
   unzip parallelmnn-binary.zip
   adb push parallelmnn-binary/* /data/local/tmp/
   ```
   3. 通过源代码构建。（需要安装一系列依赖和安卓native构建环境），之后在parallelmnn中benchmark中的
`bench_android.sh -p -64`，确保parallelmnn的可执行文件和相关的库都被push到了mobile的`/data/local/tmp/`中。

2. 执行
```shell
cd $PROJECT_HOME/mnn/src
./scripts/bench_alone.sh $model_name $mobile_name $adb_config
```
例如
```shell
./scripts/bench_alone.sh pnasnet-large huawei_p40 "-H 10.108.245.207 -P 7035"
```
脚本会自动的在device上CPU上运行1、2、4线程和GPU上，
并将CPU和GPU所有配置的集合到一起，就会生成`$PROJECT_HOME/models/${model_name}/${mobile_name}/${mobile_name}-${model_name}-layerwise-latency.csv`文件。其中


## 求解器
目前我们分别支持基于整数线性规划（ILP）的求解器和基于启发式算法的求解器。由于基于整数线性规划对求解的规模有限制，因此我们设计了基于子图划分的算法。目前子图划分支持Inception系列和Pnasnet系列的。

执行ILP的线性规划求解器：
```shell
./scripts/bench_lp.sh $model_name $mobile_name $thread $adb_config
```
例如
```shell
./scripts/bench_lp.sh pnasnet-large huawei_p40 2 "-H 10.108.245.207 -P 7035"
```
或者执行基于启发式的求解器：
```shell
./scripts/bench_greedy.sh pnasnet-large huawei_p40 2
```
调度器最后会输出理论的
之后，求解器直接把生成的执行计划push到手机的`/data/local/tmp/`目录中。

将脚本文件传输到mobile上
```
adb push scripts/set_env.sh /data/local/tmp/
adb push scripts/grun_mnn.sh /data/local/tmp/
adb push scripts/run_mnn.sh /data/local/tmp/
```

在设备上测试性能：
```shell
cd /data/local/tmp
source set_env.sh
```

```
./run_mnn.sh $model_name $thread_number
./grun_mnn.sh $model_name $thread_number
```
其中run_mnn.sh为测量ILP生成的执行计划的实际性能，
grun_mnn.sh为测量启发式生成的执行计划的实际性能。

## 最终性能对比
执行完以上流程后，
这里以huawei_p40为例，对比单独使用GPU，单独使用CPU，和使用HOPE-Scheduler调度器CPU+GPU的性能。
在host上执行以下命令：
```shell
./scripts/compare.sh $model_name $thread $adb_config
```
例如
```shell
./scripts/compare.sh pnasnet-large 2 -H 10.108.245.207 -P 7035
```
分别会打印出三种配置下的avg = xxx ms。通过对比时间即可。
## 联系
有问题欢迎联系xiachunwei@ict.ac.cn