import re

# float_reg = r'(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)? [a-z]'
# line1 = "tt - t_2_5 - 500.000000 s_2_5 + 0.000000 s_2_7 > -492.692883"
# matched_str =  re.finditer(float_reg, line1)
# for ms in matched_str:
#     old_str = ms.group()
#     new_str = old_str.replace(' ', ' * ')
#     line1 = line1.replace(old_str, new_str)

# print(line1)

# line = 'tt - t_2_2 - 500.000000 * 0.0 + 0.000000 * 0 > -490.100215'
# float_reg = r'(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)? \* (\d+)?'
# matched_str =  re.finditer(float_reg, line)
# for ms in matched_str:
#     print(ms.group())
#     print(eval(ms.group()))

a = 'abc'
line = 'aaaabbccc'
it = re.finditer(a, line)
for i in it:
    print(i.group())
    print(len(i.group()))

reg = r'([+-]( )?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)? ([+-]( )?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
line = 't_1_5 - t_1_1 + 0.0 - 500.0 + 0.0 > 3.487100'
line = 't_2_4 - t_2_5 - 500.0 - 0.0 + 0.0 > -492.692883'
line = 't_1_2 - t_1_3 - 0.0 - 0.0 + 0.0 > -495.730300'
for i in range(5):
    matched_str =  re.finditer(reg, line)
    for ms in matched_str:
        if eval(ms.group()) > 0 or str(float(eval(ms.group()))) != '-0.0':
            line = line.replace(ms.group(), '+' + str(float(eval(ms.group()))))
        else:
            line = line.replace(ms.group(), str(float(eval(ms.group()))))
    print(line)

line = 't_1_2 - t_1_3 - 0.0 - 0.0 + 0.0 > -495.730300'

def strip_zero(lines):
    new_lines = []
    reg = r'([+-]( )?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
    for line in lines:
        matched_str =  re.finditer(reg, line)
        for ms in matched_str:
            if eval(ms.group()) == 0.0:
                line = line.replace(ms.group(), "")
        new_lines.append(line)
    return new_lines

print(strip_zero([line]))

# print(matched_str.group())
