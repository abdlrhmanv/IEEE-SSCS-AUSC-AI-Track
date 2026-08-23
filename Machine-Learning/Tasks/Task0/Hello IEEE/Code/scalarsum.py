def scalarsum(scalar,arr):
    C = []
    for i in range(len(arr)):
        C.append([])
        for j in range(len(arr[0])):
            C[i].append(arr[i][j] + scalar)
    return C
