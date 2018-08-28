import multiprocessing
from multiprocessing import Queue

def printer(in_queue):
    while True:
        item = in_queue.get()
        print(item)
        if in_queue.empty():
            return

def adder(in_queue, out_queue):
    while True:
        item = in_queue.get()
        out_queue.put(item + 1)
        if in_queue.empty():
            return

def init(out_queue):
    for i in range(10):
        out_queue.put(i)

def main():
    q = [Queue() for _ in range(2)]
    p1 = Process(target=init, args=(q[0],))
    p2 = Process(target=adder, args=(q[0],q[1]))
    p3 = Process(target=printer, args=(q[1],q[2]))

    p1.start()
    p2.start()
    p3.start()

    for i in q:
        i.close()
        i.join_thread()

    p1.join()
    p2.join()
    p3.join()
