# Importing the matplotlb.pyplot 
import matplotlib.pyplot as plt 
import json
import os

def draw_gantt(cpu_data, gpu_data, convert_data, fig_name):
    print("draw_gantt")
    print(cpu_data, gpu_data, convert_data)
    # Declaring a figure "gnt" 
    fig, gnt = plt.subplots(figsize=(10, 4)) 
    print(fig_name)
    # Setting Y-axis limits 
    gnt.set_ylim(0, 20)

    def max_end_point(data):
        if data == None or len(data) == 0:
            return 0
        return max([s+d for (s, d) in data])
    
    x_limit = max([max_end_point(cpu_data), max_end_point(gpu_data), max_end_point(convert_data)])
    
    # Set font family
    font_size_num = 18
    font_times_new_roman = {'fontname':'Times New Roman'}
    # Setting X-axis limits
    # gnt.set_xlim(0, x_limit * 1.2)
    gnt.set_xlim(0, 160)
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Millisecond since start', fontsize=font_size_num, **font_times_new_roman)
    gnt.set_ylabel('Computing Devices', fontsize=font_size_num, **font_times_new_roman) 
    gnt.set_title(os.path.basename(fig_name), fontsize=font_size_num, **font_times_new_roman)
    # Setting ticks on y-axis 
    gnt.set_yticks([5, 10, 15])
    # Labelling tickes of y-axis 
    gnt.set_yticklabels(['CPU', 'GPU', 'COMM'], fontsize=font_size_num, **font_times_new_roman)

    # Setting graph attribute 
    gnt.grid(True)

    print(cpu_data)
    print(gpu_data)
    print(convert_data)
    # Declaring multiple bars in at same level and same width 
    bar_height = 4
    gnt.broken_barh(cpu_data, (3, bar_height), 
                            facecolors ='#ef8e2c', edgecolor='#ffffff')
    
    gnt.broken_barh(gpu_data, (8, bar_height), 
                                    #facecolors =('#386bec'), edgecolor='#ffffff')
                                    facecolors =('#4dad5b'), edgecolor='#ffffff')
    # Declaring a bar in schedule 
    gnt.broken_barh(convert_data, (13, bar_height), facecolors =('tab:red'), edgecolor='#ffffff')

    # plt.show()
    plt.savefig(fig_name+".pdf") 


