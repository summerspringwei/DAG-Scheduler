def func(a=None, b=None):
    if a != None:
        print("A not None")

class DeviceType:
    CPU = 0
    GPU = 3

if __name__ == "__main__":
    func("a")
    print(DeviceType.CPU == 0)
    print(DeviceType.GPU == 3)