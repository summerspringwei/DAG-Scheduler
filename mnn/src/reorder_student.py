
class Student:
    def __init__(self):
        self.index = ''
        self.student_number = ''
        self.name = ''
        self.sex = ''
        self.id_card = ''
        self.level = ''
        self.major = ''
        self.phone_number = ''
    
    def __str__(self):
        # str_student = "%s,%s,%s,%s,%s,%s,%s" % \
        #     (self.index, self.student_number, self.name, self.sex, self.id_card, self.level, self.major)
        str_student = "%s,%s,%s,%s,%s,%s" % \
            (self.student_number, self.name, self.sex, self.id_card, self.major, self.level)
        return str_student


def read_org(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    stu_infos = []
    for line in lines:
        com = line.strip().split('\t')
        stu = []
        for c in com:
            if c!='' and c!=' ':
                stu.append(c.strip())
        stu_infos.append(stu)
    return stu_infos


def main():
    orginal_order_file_path = 'original_order.txt'
    stu_infos = read_org(orginal_order_file_path) 
    # for stu in stu_infos:
    #     print(stu)
    # print('-*'*20)
    meta_student_file_path = 'meta_student_info.txt'
    meta_infos = read_org(meta_student_file_path)
    name_stu_map = {}
    for meta in meta_infos:
        assert(len(meta) == 7)
        name = meta[2]
        s = Student()
        s.name = name
        s.id_card = meta[3]
        s.level = meta[4]
        s.major = meta[5]
        s.phone_number = meta[6]
        name_stu_map[name] = s
        # print(meta)
    for stu in stu_infos:
        assert(len(stu) == 4)
        name = stu[1]
        s = name_stu_map[name]
        s.index = stu[0]
        s.sex = stu[2]
        s.student_number = stu[3]
        print(s)
    
        

if __name__ == "__main__":
    main()