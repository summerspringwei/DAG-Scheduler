
import re

line1 = 'cell_stem_0/cell_output/concat'
line3 = 'cell_0/cell_output/concat'
line2 = 'cell_0/comb_iter_2/combine/add___tr4cell_0/cell_output/concat'
concat_reg = 'cell_(stem_)?[0-9]+/cell_output/concat'
matched_str =  re.finditer(concat_reg, line1)
for ms in matched_str:
    print(ms.group()==line1)

matched_str = re.finditer(concat_reg, line2)
for ms in matched_str:
    print(ms.group()==line2)

matched_str =  re.finditer(concat_reg, line3)
for ms in matched_str:
    print(ms.group()==line3)

al = [1,2,3,4]
bl = [2,3,4,5,6]
a = set()
b = set()
for n in al:
    a.add(n)
for n in bl:
    b.add(n)
c = (a.intersection(b))
a.difference_update(c)
b.difference_update(c)

class NamedList(list):
    def __init__(self, name):
        self.name = name

l = list()
nl = NamedList('a')
print(isinstance(l, NamedList))
print(isinstance(nl, NamedList))

def append(arr):
    arr.append('0')
    print(arr)


def add(num):
    num += 1
    print(num)


al = ['1']
print(al)
append(al)
print(al)
num = 1
print(num)
add(num)
print(num)

# All subgraph:
# name: cell_17/comb_iter_0/
# latency: (34.894189,8.365660,5.638775,5.443746)
# nodes:['cell_17/comb_iter_0/left/Relu', 'cell_17/comb_iter_0/left/separable_5x5_1/separable_conv2d/depthwise', 'cell_17/comb_iter_0/left/separable_5x5_1/separable_conv2d', 'cell_17/comb_iter_0/left/separable_5x5_2/separable_conv2d/depthwise', 'cell_17/comb_iter_0/left/separable_5x5_2/separable_conv2d', 'cell_17/comb_iter_0/right/Relu', 'cell_17/comb_iter_0/right/separable_3x3_1/separable_conv2d/depthwise', 'cell_17/comb_iter_0/right/separable_3x3_1/separable_conv2d', 'cell_17/comb_iter_0/right/separable_3x3_2/separable_conv2d/depthwise', 'cell_17/comb_iter_0/right/separable_3x3_2/separable_conv2d', 'cell_17/comb_iter_0/combine/add']
# parents:{'cell_17/1x1/Conv2D', 'cell_17/subgraph_0', 'cell_17/subgraph_1', 'cell_17/prev_1x1/Conv2D'}
# children:{'cell_17/cell_output/concat', 'cell_17/subgraph_0'}
# input_tensors[('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396)]
# output_tensors[['1,672,11,11']]

# name: cell_17/comb_iter_1/
# latency: (34.808156,8.264269,5.638775,5.443746)
# nodes:['cell_17/comb_iter_1/left/Relu', 'cell_17/comb_iter_1/left/separable_5x5_1/separable_conv2d/depthwise', 'cell_17/comb_iter_1/left/separable_5x5_1/separable_conv2d', 'cell_17/comb_iter_1/left/separable_5x5_2/separable_conv2d/depthwise', 'cell_17/comb_iter_1/left/separable_5x5_2/separable_conv2d', 'cell_17/comb_iter_1/right/Relu', 'cell_17/comb_iter_1/right/separable_3x3_1/separable_conv2d/depthwise', 'cell_17/comb_iter_1/right/separable_3x3_1/separable_conv2d', 'cell_17/comb_iter_1/right/separable_3x3_2/separable_conv2d/depthwise', 'cell_17/comb_iter_1/right/separable_3x3_2/separable_conv2d', 'cell_17/comb_iter_1/combine/add']
# parents:{'cell_17/subgraph_0', 'cell_17/prev_1x1/Conv2D'}
# children:{'cell_17/cell_output/concat', 'cell_17/subgraph_0'}
# input_tensors[('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396)]
# output_tensors[['1,672,11,11']]

# name: cell_17/comb_iter_2/
# latency: (0.662500,1.546872,8.458162,8.165618)
# nodes:['cell_17/comb_iter_2/left/AvgPool2D/AvgPool', 'cell_17/comb_iter_2/left/AvgPool2D/AvgPool___tr4cell_17/comb_iter_2/combine/add', 'cell_17/comb_iter_2/combine/add', 'cell_17/comb_iter_2/combine/add___tr4cell_17/cell_output/concat']
# parents:{'cell_17/1x1/Conv2D', 'cell_17/subgraph_0', 'cell_17/subgraph_1', 'cell_17/prev_bn/FusedBatchNormV3___tr4cell_17/comb_iter_2/combine/add'}
# children:{'cell_17/cell_output/concat', 'cell_17/subgraph_0'}
# input_tensors[('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396)]
# output_tensors[['1,672,11,11']]

# name: cell_17/comb_iter_3/
# latency: (1.151978,2.287033,5.638775,5.443746)
# nodes:['cell_17/comb_iter_3/left/AvgPool2D/AvgPool', 'cell_17/comb_iter_3/right/AvgPool2D/AvgPool', 'cell_17/comb_iter_3/left/AvgPool2D/AvgPool___tr4cell_17/comb_iter_3/combine/add', 'cell_17/comb_iter_3/right/AvgPool2D/AvgPool___tr4cell_17/comb_iter_3/combine/add', 'cell_17/comb_iter_3/combine/add', 'cell_17/comb_iter_3/combine/add___tr4cell_17/cell_output/concat']
# parents:{'cell_17/subgraph_0', 'cell_17/prev_1x1/Conv2D'}
# children:{'cell_17/cell_output/concat', 'cell_17/subgraph_0'}
# input_tensors[('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396)]
# output_tensors[['1,672,11,11']]

# name: cell_17/comb_iter_4/
# latency: (16.943800,4.236519,8.458162,8.165618)
# nodes:['cell_17/comb_iter_4/left/Relu', 'cell_17/comb_iter_4/left/separable_3x3_1/separable_conv2d/depthwise', 'cell_17/comb_iter_4/left/separable_3x3_1/separable_conv2d', 'cell_17/comb_iter_4/left/separable_3x3_2/separable_conv2d/depthwise', 'cell_17/comb_iter_4/left/separable_3x3_2/separable_conv2d', 'cell_17/comb_iter_4/combine/add']
# parents:{'cell_17/1x1/Conv2D', 'cell_17/subgraph_1'}
# children:{'cell_17/cell_output/concat', 'cell_17/subgraph_0'}
# input_tensors[('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396)]
# output_tensors[['1,672,11,11']]

# name: cell_17/subgraph_0
# latency: (47.394600,9.203122,20.601522,20.196405)
# nodes:['cell_17/Relu', 'cell_17/prev_1x1/Conv2D', 'cell_17/cell_output/concat', 'cell_17/prev_bn/FusedBatchNormV3___tr4cell_17/comb_iter_2/combine/add']
# parents:{'cell_17/comb_iter_3/combine/add___tr4cell_17/cell_output/concat', 'cell_17/comb_iter_4/', 'cell_17/comb_iter_1/', 'cell_17/comb_iter_2/combine/add___tr4cell_17/cell_output/concat', 'cell_17/comb_iter_0/combine/add', 'cell_17/comb_iter_1/combine/add', 'cell_17/comb_iter_3/', 'cell_17/comb_iter_0/', 'cell_17/comb_iter_2/', 'cell_15/cell_output/concat', 'cell_17/comb_iter_4/combine/add'}
# children:{'cell_17/comb_iter_1/left/Relu', 'final_layer/Relu', 'cell_17/comb_iter_1/', 'cell_17/comb_iter_3/left/AvgPool2D/AvgPool', 'cell_17/comb_iter_3/', 'cell_17/comb_iter_0/', 'cell_17/comb_iter_2/', 'cell_17/comb_iter_0/right/Relu', 'cell_17/comb_iter_1/right/Relu', 'cell_17/comb_iter_3/right/AvgPool2D/AvgPool', 'cell_17/comb_iter_2/combine/add'}
# input_tensors[('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396), ('1,672,11,11', 2.819387396), ('1,4032,11,11', 3.6851981129999998)]
# output_tensors[['1,4032,11,11']]

# name: cell_17/subgraph_1
# latency: (46.985900,8.201300,3.685198,3.865168)
# nodes:['cell_17/Relu_1', 'cell_17/1x1/Conv2D']
# parents:{'cell_16/cell_output/concat'}
# children:{'cell_17/comb_iter_4/', 'cell_17/comb_iter_0/left/Relu', 'cell_17/comb_iter_4/left/Relu', 'cell_17/comb_iter_0/', 'cell_17/comb_iter_2/', 'cell_17/comb_iter_2/left/AvgPool2D/AvgPool', 'cell_17/comb_iter_4/combine/add'}
# input_tensors[('1,4032,11,11', 3.6851981129999998)]
# output_tensors[['1,4032,11,11']]

# cell_17/comb_iter_0/ 1 Operator latency: 34.894189 8.365660 5.443746 5.638775
# cell_17/comb_iter_1/ 2 Operator latency: 34.808156 8.264269 5.443746 5.638775
# cell_17/comb_iter_2/ 3 Operator latency: 0.662500 1.546872 8.165618 8.458162
# cell_17/comb_iter_3/ 4 Operator latency: 1.151978 2.287033 5.443746 5.638775
# cell_17/comb_iter_4/ 5 Operator latency: 16.943800 4.236519 8.165618 8.458162
# cell_17/subgraph_0 6 Operator latency: 47.394600 9.203122 20.196405 20.601522
# cell_17/subgraph_1 7 Operator latency: 46.985900 8.201300 3.865168 3.685198