模型来源：
mnist类网络：1x1x28x28数据经过卷积层后，接全连接层加激活函数，接全连接，后接softmax，输出1x10
另外两个分支测试用，1个分支接argmax，1个分支接resize

1.双击Netron Setup 20210106(HiAI特别版).exe安装Netron可视化om模型
2.View->Show Names可以显示算子名称，双击算子可查看算子属性
该模型属于aicpu的算子：
ResizeBilinearV2，ArgMaxV2，SoftmaxV2，Relu6，BiasAdd，Cast
该模型属于aicore的算子：
Conv2D，MatMul，Add，AscendQuant，AscendDequant
3.算子融合
除MatMulBiasAddFusionPass，NotRequantFusionPass外，算子已经去除融合

说明：
1.由于算子原型限制与融合规则ge构图的限制，以下两个是融合的算子，否则报错
MatMulBiasAddFusionPass
NotRequantFusionPass
2.由于算子原型的限制需要TranData算子将NCHW转换为NC1HWC0
3.由于aicpu算子与aicore算子原型与实现的限制，模型中已经将能使用aicpu的算子改为使用aicpu算子
4.profiling数据为运行50次纯推理的结果
