
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

if __name__ == "__main__":
  s1 = Student("s1", 10)
  s2 = Student("s2", 11)
  l = list()
  l.append(s1)
  l.append(s2)
  update_age(l)
  for s in l:
    print(s.age)
  num = 1
  update_int(num)
  print(num)
  print(max(1,2))
