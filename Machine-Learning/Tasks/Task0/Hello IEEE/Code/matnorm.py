import math

def matnorm(arr):
    total_sum = 0  
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            total_sum += arr[i][j] ** 2  
            
    return math.sqrt(total_sum)  