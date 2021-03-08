模型来源：
1个大分支取自inceptionv3网络的一部分
另1个分支测试用接了resize算子

1.双击Netron Setup 20210106(HiAI特别版).exe安装Netron可视化om模型
2.View->Show Names可以显示算子名称，双击算子可查看算子属性
该模型属于aicpu的算子：
Relu，Cast
其它属于aicore算子
3.算子融合
除PadV3FusionPass，ResizeFusionPass外，算子已经去除融合

说明：
1.由于算子原型限制与融合规则ge构图的限制，以下两个是融合的算子，否则报错
PadV3FusionPass
ResizeFusionPass
2.由于算子原型的限制需要TranData算子将NCHW转换为NC1HWC0
3.由于aicpu算子与aicore算子原型与实现的限制，模型中已经将能使用aicpu的算子改为使用aicpu算子
4.profiling数据为运行50次纯推理的结果
