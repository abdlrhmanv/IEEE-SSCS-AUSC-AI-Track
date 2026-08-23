def matmul(arr1,arr2):
    C = []
    if len(arr1[0]) != len(arr2):
        return "Error: Dimension mismatch"
    for i in range(len(arr1)):
        C.append([])
        for j in range(len(arr2[0])):
            C[i].append(0)
            for k in range(len(arr2)):
                C[i][j] += arr1[i][k] * arr2[k][j]
    return C