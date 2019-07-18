
class Student:
  def __init__(self, name, age):
    self.name = name
    self.age = age

def update_age(l):
  for s in l:
    s.age += 1

def update_int(num):
  num += 1
  print(num)


def get_ages(l):
  ages = list()
  for s in l:
    s_new = s
    ages.append(s_new)
  return ages

def key_sort_student(stu):
  return stu.age

if __name__ == "__main__":
  s1 = Student("s1", 10)
  s2 = Student("s2", 11)
  s3 = Student("s3", 8)
  l = list()
  l.append(s1)
  l.append(s2)
  l.append(s3)
  l.sort(key=key_sort_student)
  for s in l:
    print(s.name + str(s.age))
  print("Sort done.")
  update_age(l)
  for s in l:
    print(s.age)
  num = 1
  update_int(num)
  print(num)
  print(max(1,2))
  ages = get_ages(l)
  for s in ages:
    print(s.name + str(s.age))
  