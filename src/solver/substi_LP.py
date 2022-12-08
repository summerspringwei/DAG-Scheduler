import re

def read_lines(file_path):
    f = open(file_path, 'r')
    lines =  f.readlines()
    f.close()
    return lines


# For pnasnet-mobile cpu alone
def initialize_s_variable_map():    
    s_variable_map = {}
    for i in range(1, 9):
        s_variable_map['s_1_%d' % (i)] = '1'
        s_variable_map['s_2_%d' % (i)] = '0'
    # s_variable_map['s_1_5'] = '0'
    # s_variable_map['s_2_5'] = '1'
    return s_variable_map


def parse_u_variable_map(result_file_path):
    f = open(result_file_path, 'r')
    u_variable_map = {}
    lines = f.readlines()
    for line in lines:
        # Find lines with s_
        line = line.strip()
        if line.find('u_') >= 0:
            com = line.split(' ')
            striped_com = []
            for c in com:
                if c != '':
                    striped_com.append(c)
            
            if len(striped_com) == 6:
                # print(striped_com)
                u_variable_map[striped_com[1]] = striped_com[3]
    return u_variable_map


def insert_multi_sign(lines):
    float_reg = r'(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)? [a-z]'
    new_lines = []
    for line in lines:
        matched_str =  re.finditer(float_reg, line)
        for ms in matched_str:
            old_str = ms.group()
            new_str = old_str.replace(' ', ' * ')
            line = line.replace(old_str, new_str)
            
        new_lines.append(line)
    return new_lines


def replace_variables(lines, variable_map):
    new_lines = []
    for line in lines:
        for k, v in variable_map.items():
            if line.find(k) >= 0:
                line = line.replace(k, v)
        new_lines.append(line)
    return new_lines


def eval_multiply_expression(lines):
    new_lines = []
    reg = r'(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)? \* (\d+)?'
    for line in lines:
        matched_str =  re.finditer(reg, line)
        for ms in matched_str:
            line = line.replace(ms.group(), str(float(eval(ms.group()))))
        new_lines.append(line)
    return new_lines


def eval_add_expression(lines):
    new_lines = []
    reg = r'([+-]( )?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)? ([+-]( )?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
    for line in lines:
        for i in range(3):
            matched_str =  re.finditer(reg, line)
            for ms in matched_str:
                if eval(ms.group()) > 0 or (float(eval(ms.group()))==0.0 and str(float(eval(ms.group()))) != '-0.0'):
                    line = line.replace(ms.group(), '+' + str(float(eval(ms.group()))))
                else:
                    line = line.replace(ms.group(), str(float(eval(ms.group()))))
        new_lines.append(line)
    return new_lines


def strip_zero_expression(lines):
    new_lines = []
    reg = r'([+-]( )?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
    for line in lines:
        if line.find('>') <=0:
            continue
        com = line.split('>')
        matched_str = re.finditer(reg, com[0])
        for ms in matched_str:
            if eval(ms.group()) == 0.0:
                line = line.replace(ms.group(), "")
        new_lines.append(line)
    return new_lines


def move_to_right(lines):
    new_lines = []
    reg = r'([+-]( )?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
    for line in lines:
        if line.find('>') < 0:
            continue
        has_left = False
        has_right = False
        com = line.split('>')
        matched_str =  re.finditer(reg, com[0])
        for ms in matched_str:
            has_left = True
            left_number = eval(ms.group())
            com[0] = com[0].replace(ms.group(), "")
        matched_str = re.finditer(reg, com[1])
        for ms in matched_str:
            has_right = True
            right_number = eval(ms.group())
        if has_left and has_right:
            # print("%f %f" %(left_number, right_number))
            com[1] = str(-left_number + right_number)
            line = com[0] + ' > ' + com[1]
            new_lines.append(line)
        elif has_left and not has_right:
            print("Error!")
        elif not has_left and has_right:
            new_lines.append(line)
    return new_lines


def strip_always_true(lines):
    new_lines = []
    reg = r'([+-]( )?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
    for line in lines:
        matched_str = re.finditer(reg, line)
        for ms in matched_str:
            right_number = eval(ms.group())
            if right_number < -400:
                print("strip %s" % (line.strip()))
                continue
            new_lines.append(line)
    return new_lines


def build_constraints_relationship(lines):
    constraint_map = {}
    for line in lines:
        com = line.split('>')
        assert(len(com) > 1)
        num = float(eval(com[1].strip()))
        vs = com[0].split('-')
        if len(vs) < 2:
            continue
        left_variable = vs[0].strip()
        right_variable = vs[1].strip()
        constraint_map[(left_variable, right_variable)] = num
    return constraint_map


def reasoning_constrains(constraint_map):
    while True:
        new_constrains = []
        for k1, v1 in constraint_map.items():
            for k2, v2 in constraint_map.items():
                if k1 == k2:
                    if v1 != v2:
                        print(">>>> Dulplicate constraints %s %s %f %f <<<<<" % (k1, k2, v1, v2))
                    continue
                k1_left = k1[0]
                k1_right = k1[1]
                k2_left = k2[0]
                k2_right = k2[1]
                # x - y > a & y - z > c => x - z > (a+c)
                if k1_right == k2_left and (k1_left, k2_right) not in constraint_map.keys() \
                    and ((k1_left, k2_right), (v1 + v2)) not in new_constrains:
                    new_constrains.append(((k1_left, k2_right), (v1 + v2)))
                    print("%s - %s > %f & %s - %s > %f => %s - %s > %f" \
                        % (k1_left, k1_right, v1, k2_left, k2_right, v2, k1_left, k2_right, (v1 + v2)))
                # x - y > a & z - x > c => z - y > (a+c)
                elif k1_left == k2_right and (k2_left, k1_right) not in constraint_map.keys() \
                    and ((k2_left, k1_right), (v1 + v2)) not in new_constrains:
                    new_constrains.append(((k2_left, k1_right), (v1 + v2)))
                    print("%s - %s > %f & %s - %s > %f => %s - %s > %f" \
                        % (k1_left, k1_right, v1, k2_left, k2_right, v2, k2_left, k1_right, (v1 + v2)))
                # Constraints corruption!  x - y > a & y - x > c
                if k1_left == k2_right and k1_right == k2_left:
                    if v1 >= 0 and v2 >= 0:
                        print("Constraints corruption detected! %s %s %f %f" % (k1_left, k1_right, v1 ,v2))
                        return False
        if len(new_constrains) == 0:
            break
        else:
            for cons in new_constrains:
                constraint_map[cons[0]] = cons[1]
    for k, v in constraint_map.items():
        print("%s - %s > %f" %(k[0], k[1], v))
    return True


def main():
    file_path = '../models/pnasnet-mobile/lenovo_k5/subgraphs-cell_0.lp'
    lines = insert_multi_sign(read_lines(file_path))
    # u_variable_map = parse_u_variable_map('../models/pnasnet-mobile/lenovo_k5/lp-result-subgraphs-cell_0.txt')
    u_variable_map = parse_u_variable_map('../models/pnasnet-mobile/lenovo_k5/lp-result-cp_alone_subgraphs-cell_0.txt')
    s_variable_map = initialize_s_variable_map()
    variable_map = {}
    
    for k, v in u_variable_map.items():
        variable_map[k] = v
    for k, v in s_variable_map.items():
        variable_map[k] = v
    print(variable_map)

    new_lines = replace_variables(lines, variable_map)
    for line in new_lines:
        print(line.strip())
    print("Replace variables")
    new_lines = eval_multiply_expression(new_lines)
    # for line in new_lines:
    #     print(line.strip())
    new_lines = eval_add_expression(new_lines)
    # for line in new_lines:
    #     print(line.strip())
    
    new_lines = strip_zero_expression(new_lines)
    # for line in new_lines:
    #     print(line.strip())
    new_lines = move_to_right(new_lines)
    for line in new_lines:
        print(line.strip())
    print("Eval expression")
    new_lines = strip_always_true(new_lines)
    for line in new_lines:
        print(line.strip())
    constraint_map = build_constraints_relationship(new_lines)
    for k, v in constraint_map.items():
        print("%s %s" % (k, v))
    print(reasoning_constrains(constraint_map))

if __name__ == "__main__":
    main()
