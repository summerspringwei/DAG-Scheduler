
from visualization import *
# 1+2+3+5=11
def draw_my_timeline():
    cpu_data = [(0.0, 2.2553), (2.2553, 9.756200000000002), (12.0115, 15.7561), (27.7676, 23.8447), (54.3072, 22.4969), (76.8041, 0.8492999999999999)] 
    gpu_data = [(0.0, 8.4686), (8.4686, 12.8522), (21.3208, 15.447299999999998), (36.7681, 17.539099999999998), (54.3072, 15.6462), (71.026, 5.7781)] 
    convert_data = [(76.8041, 1.863480583), (78.60409999999999, 2.160795796), (80.6534, 1.995379747)]
    offset = 2
    gpu_data = [(start+offset, latency) for (start, latency) in gpu_data]
    convert_data = [(start+offset, latency) for (start, latency) in convert_data]
    convert_data.append((0, offset))
    draw_gantt(cpu_data, gpu_data, convert_data, "Our Approach")


def draw_mosaic_timeline():
    cpu_latency = [x / 1000 for x in [2255.3, 6427.7]]
    gpu_latency = [x / 1000 for x in  [15646.2, 8020.3, 15551.8, 19375.1, 8468.6, 12852.2, 15447.3, 17539.1, 19780.5]]
    convert_latency = [1, 0.5]
    start = 2
    cpu_data = []
    gpu_data = []
    convert_data = []
    for latency in cpu_latency:
        cpu_data.append((start, latency))
        start += latency
    for latency in gpu_latency:
        gpu_data.append((start, latency))
        start += latency
    convert_data.append((start, 1))
    convert_data.append((0, 2))
    draw_gantt(cpu_data, gpu_data, convert_data, "MOSAIC: Serial Execution")



if __name__=="__main__":
    draw_my_timeline()
    draw_mosaic_timeline()