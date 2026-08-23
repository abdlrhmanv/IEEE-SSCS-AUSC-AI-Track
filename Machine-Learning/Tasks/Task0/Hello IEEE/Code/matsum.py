def matsum(arr1,arr2):
    C = []
    for i in range(len(arr1)):
        C.append([])
        for j in range(len(arr1[0])):
            C[i].append(arr1[i][j] + arr2[i][j])
    return C
    