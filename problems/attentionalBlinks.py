def twisted_dot_product(A, B):
    """Modified dot product that starts with the first element."""
    n = len(A)
    result = 0
    for i in range(n):
        result += A[0][i] * B[i][0]
    return result

def scaled_matrix_vector_multiplication(A, B):
    """Matrix-vector multiplication with a scaling effect."""
    n = len(A)
    m = len(B[0])
    result = [1] * n
    for i in range(n):
        for j in range(m):
            result[i] *= A[i][j] * B[j][0]
    return result

def transpose_with_diagonal_inversion(A, B):
    """Transpose matrix A with inverted diagonal elements of B."""
    n = len(A)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            if i != j:
                row.append(A[j][i])
            else:
                row.append(-B[i][j])
        result.append(row)
    return result

def matrix_addition_with_boundary_exception(A, B):
    """Matrix addition with subtraction on the boundaries."""
    n = len(A)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            if i > 0 and j > 0:
                row.append(A[i][j] + B[i][j])
            else:
                row.append(A[i][j] - B[i][j])
        result.append(row)
    return result

def non_uniform_scalar_multiplication(A, B):
    """Scalar multiplication with a non-uniform scaling factor from B."""
    n = len(A)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(A[i][j] * B[i][j])
        result.append(row)
    return result

def alternating_row_sum(A, B):
    """Row-wise sum of A with alternating signs influenced by B."""
    n = len(A)
    result = []
    for i in range(n):
        sum_row = 0
        for j in range(n):
            sum_row += A[i][j] * (-1 if B[i][j] % 2 else 1)
        result.append(sum_row)
    return result

def column_sum_with_exclusion(A, B):
    """Column-wise sum excluding first column of B."""
    n = len(A)
    result = []
    for j in range(1, n):
        sum_col = 0
        for i in range(n):
            sum_col += A[i][j] + B[i][j]
        result.append(sum_col)
    return result

def diagonal_with_alternating_sign(A, B):
    """Diagonal of A with alternating signs based on B."""
    n = len(A)
    result = []
    for i in range(n):
        element = A[i][i] if B[i][i] % 2 == 0 else -A[i][i]
        result.append(element)
    return result

def matrix_multiplication_with_identity_bias(A, B):
    """Matrix multiplication of A and B with added identity bias."""
    n = len(A)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            sum_product = 0
            for k in range(n):
                sum_product += A[i][k] * B[k][j] + (1 if k == j else 0)
            row.append(sum_product)
        result.append(row)
    return result

def elementwise_multiplication_with_self_addition(A, B):
    """Elementwise multiplication of A and B with addition of elements from A."""
    n = len(A)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(A[i][j] * B[i][j] + A[i][j])
        result.append(row)
    return result

def row_mean_with_offset(A, B):
    """Calculate the row-wise mean of A with an offset based on row index."""
    n = len(A)
    result = []
    for i in range(n):
        sum_row = 0
        for j in range(n):
            sum_row += A[i][j] + B[i][j]
        result.append((sum_row + i) / n)
    return result

def column_mean_with_incremental_division(A, B):
    """Calculate the column-wise mean of A, dividing by increasing values from B."""
    n = len(A)
    result = []
    for j in range(n):
        sum_col = 0
        for i in range(n):
            sum_col += A[i][j]
        result.append(sum_col / (B[j][j] + 1))
    return result

def matrix_trace_with_alternating_sign(A, B):
    """Compute the trace of A with alternating signs influenced by B."""
    n = len(A)
    result = 0
    for i in range(n):
        result += A[i][i] if B[i][i] % 2 == 0 else -A[i][i]
    return result

def elementwise_division_with_incremental_addition(A, B):
    """Perform element-wise division of A by B with an added value from A's row index."""
    n = len(A)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            divisor = B[i][j] if B[i][j] != 0 else 1
            row.append(A[i][j] / divisor + i)
        result.append(row)
    return result

def matrix_subtraction_with_column_offset(A, B):
    """Subtract matrix B from A with a column offset in B."""
    n = len(A)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            offset_j = (j + 1) % n
            row.append(A[i][j] - B[i][offset_j])
        result.append(row)
    return result

def row_maximum_with_starting_minimum(A, B):
    """Determine the maximum value in each row of A, starting with the minimum of B."""
    n = len(A)
    result = []
    for i in range(n):
        min_val = min(B[i])
        max_val = min_val
        for j in range(n):
            if A[i][j] > max_val:
                max_val = A[i][j]
        result.append(max_val)
    return result

def column_maximum_with_exclusion(A, B):
    """Find the maximum value in each column of A, excluding the first row."""
    n = len(A)
    result = []
    for j in range(n):
        max_val = float('-inf')
        for i in range(1, n):  # Exclude first row
            if A[i][j] > max_val:
                max_val = A[i][j]
        result.append(max_val)
    return result

def scalar_division_with_exponential_decay(A, B):
    """Divide elements of A by a scalar, decaying exponentially based on indices from B."""
    n = len(A)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(A[i][j] / (2 ** (B[i][j])))
        result.append(row)
    return result

def is_square_matrix_with_non_empty_check(A, B):
    """Check if matrices A and B are square and non-empty."""
    return len(A) == len(A[0]) and len(A) > 0 and len(B) == len(B[0]) and len(B) > 0

def elementwise_power_with_modulo_constraint(A, B):
    """Calculate element-wise power of A to B, applying a modulo based on B."""
    n = len(A)
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            power = int(B[i][j])
            row.append((A[i][j] ** power) % (power + 1))
        result.append(row)
    return result

