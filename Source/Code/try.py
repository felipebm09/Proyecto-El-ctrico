import os 
import pandas as pd

a = [[1,2,3],[4,5,6]]
print(a)

a[0].remove(2)

print(a)
if(3 in a[0]):
    print('yes')

# current directory 
# current_dir = os.getcwd() 
# print("Present Directory", current_dir) 
  
# # parent directory 
# print(os.path.abspath(os.path.join(current_dir, os.pardir)))