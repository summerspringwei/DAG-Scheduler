
import queue

s = dict()
print(len(s))

q = queue.Queue()
q.put('a')
q.put('b')
print(q.qsize())
a = q.get()
print(a)
print(q.qsize())

line = "%d s_%d_%f" % (1, 2, 3.123)
print(line)