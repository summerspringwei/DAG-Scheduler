# Importing the matplotlb.pyplot 
import matplotlib.pyplot as plt 
import json
import os

def draw_gantt(cpu_data, gpu_data, convert_data, fig_name):
    print("draw_gantt")
    print(cpu_data, gpu_data, convert_data)
    # Declaring a figure "gnt" 
    fig, gnt = plt.subplots() 
    print(fig_name)
    # Setting Y-axis limits 
    gnt.set_ylim(0, 20)

    def max_end_point(data):
        if data == None or len(data) == 0:
            return 0
        return max([s+d for (s, d) in data])
    
    x_limit = max([max_end_point(cpu_data), max_end_point(gpu_data), max_end_point(convert_data)])
    # Setting X-axis limits
    gnt.set_xlim(0, x_limit * 1.2)

    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Millisecond since start')
    gnt.set_ylabel('Processor') 
    gnt.set_title(os.path.basename(fig_name), fontsize=16)
    # Setting ticks on y-axis 
    gnt.set_yticks([5, 10, 15])
    # Labelling tickes of y-axis 
    gnt.set_yticklabels(['CPU', 'GPU', 'CONVERT'])

    # Setting graph attribute 
    gnt.grid(True)

    print(cpu_data)
    print(gpu_data)
    print(convert_data)
    # Declaring multiple bars in at same level and same width 
    gnt.broken_barh(cpu_data, (3, 4), 
                            facecolors ='#ef8e2c')
    
    gnt.broken_barh(gpu_data, (8, 4), 
                                    facecolors =('#386bec')) 
    # Declaring a bar in schedule 
    gnt.broken_barh(convert_data, (13, 4), facecolors =('tab:red')) 

    # plt.show()
    plt.savefig(fig_name) 


