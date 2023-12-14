





def function_03(A, B):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + B[i][j]
    return result


def function_04(A, s):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] * s
    return result


def function_05(A):
    result = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j]
    return result


def function_06(A):
    result = [0] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            result[j] += A[i][j]
    return result


def function_07(A):
    result = []
    for i in range(len(A)):
        result.append(A[i][i])
    return result


def function_08(A, B):
    result = [[0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


def function_09(A, B):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] * B[i][j]
    return result


def function_10(A):
    result = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j]
        result[i] /= len(A[0])
    return result

def function_11(A):
    result = 0
    for i in range(len(A)):
        result += A[i][i]
    return result


def function_12(A, B):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] / B[i][j]
    return result


def function_13(A, B):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] - B[i][j]
    return result


def function_14(A):
    result = [float('-inf')] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > result[i]:
                result[i] = A[i][j]
    return result


def function_15(A):
    result = [float('-inf')] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            if A[i][j] > result[j]:
                result[j] = A[i][j]
    return result




def function_17(A):
    return len(A) == len(A[0])

def function_19(A):
    result = [float('inf')] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] < result[i]:
                result[i] = A[i][j]
    return result


def function_20(A):
    result = [float('inf')] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            if A[i][j] < result[j]:
                result[j] = A[i][j]
    return result

def function_21(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = -A[i][j]
    return result


def function_22(A):
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > 0:
                count += 1
    return count


def function_23(A):
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] < 0:
                count += 1
    return count


def function_26(A):
    result = []
    for i in range(len(A)):
        for j in range(len(A[0])):
            result.append(A[i][j])
    return result
