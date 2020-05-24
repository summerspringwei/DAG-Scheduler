class Stu:
    def __init__(self, name, age):
        self.name = name
        self.age = age

if __name__ == "__main__":
    s1 = Stu('1',10)
    s2 = Stu('2', 11)
    print(s2.age)
    s2 = s1
    print(s2.age)
    s2.age = 9
    print(s2.age)
    print(s1.age)
    s1.age = 12
    print(s2.age)
    print(s1.age)

    a = [1,2,3]
    b=list(a)
    print(b)
    b.remove(1)
    print(b)
    print(a)
    

