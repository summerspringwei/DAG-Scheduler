from utils import *
# import read_profile_data
import read_profile_data

print(read_profile_data.CPU_thread_index)
read_profile_data.CPU_thread_index = 4
print(read_profile_data.CPU_thread_index)

model, mobile = parse_model_mobile()
print(model+mobile)
a = 10
if a>0:
    file_name = 'a'
else:
    file_name = 'b'
print(file_name)

