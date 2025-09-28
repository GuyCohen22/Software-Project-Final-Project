#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "symnmf.h"

static const double BETA = 0.5;

typedef enum { 
    STATUS_OK = 0,
    STATUS_ERROR = 1
} Status;

/* Row/column dimensions of a matrix. */
    typedef struct {
    int rows;
    int cols;
} Shape;

/* Matrix wrapper: 'data' is a rows-length array of pointers,
 * each pointing to a cols-length array of doubles (row-major).
 * Ownership: unless stated otherwise, functions that return a Matrix
 * allocate 'data' and expect the caller to free it with matrix_data_free(data, rows). */
typedef struct {
    double** data;
    Shape shape;
} Matrix;

/* Abort-from-main helper: if (result != STATUS_OK), print a generic error
 * and return 1 from the current function (int-returning, typically main). */
#define ASSERT_OK(result)                      \
  do {                                         \
    if ((result) != STATUS_OK) {               \
      printf("An Error Has Occurred\n");       \
      return 1;                                \
    }                                          \
  } while (0)

/* ======================= Matrix Memory Utilities ======================= */

/**
 * @brief Frees a dynamically allocated 2D array of doubles.
 *
 * This function releases both levels of allocation that make up a
 * row-major 2D array:
 *   1. Each individual row (data[i]).
 *   2. The top-level array of row pointers (data itself).
 *
 * @param data Pointer to an array of 'rows' row pointers. May be NULL.
 *             Each row pointer must have been allocated with malloc/calloc/realloc
 *             (or be NULL). If NULL, the function does nothing.
 * @param rows Number of valid row pointers in 'data' to free. Must not
 *             exceed the number of rows actually allocated.
 *
 * @note This function does **not** free any enclosing 'matrix' struct,
 *       only the row-pointer table and the row buffers.
 *       After this call, 'data' becomes invalid and must not be dereferenced.
 */
void matrix_data_free(double** data, int rows) {
    int i;
    if (!data) {
        return;
    }

    for (i = 0; i < rows; i++) {
        if (data[i]) free(data[i]);
    }

    free(data);
}


/**
 * @brief Allocates a dynamically sized 2D array of doubles.
 *
 * Creates a row-major 2D array by allocating a table of 'rows' pointers and,
 * for each row, a contiguous buffer of 'cols' doubles. All memory is obtained
 * with 'calloc', so both the pointer table and the elements are zero-initialized.
 *
 * @param rows Number of rows to allocate. Must be > 0.
 * @param cols Number of columns to allocate. Must be > 0.
 *
 * @return Pointer to the row pointer table ('double**') on success,
 *         or NULL on failure (any allocation fails or sizes are invalid).
 */
static double** matrix_data_alloc(int rows, int cols) {
    int i;
    double** data = (double**)calloc(rows, sizeof(double*));
    if (!data) {
        return NULL;
    }
    
    for (i = 0; i < rows; i++) {
        data[i] = (double*)calloc(cols, sizeof(double));
        if (!data[i]) {
            matrix_data_free(data, i);
            return NULL;
        }
    }

    return data;
}

/**
 * @brief Frees each matrix in an array of dynamically allocated 2D double arrays.
 *
 * Given an array of matrix row-pointer tables ('double**'), this function calls
 * 'matrix_data_free' on every non-NULL entry. It assumes each matrix is a
 * row-major 2D array with exactly 'rows' rows.
 *
 * @param matrices Array of pointers to matrices ('double**').
 * @param num_of_matrices Number of entries in 'matrices'. 
 * @param rows Number of rows to free in each matrix.
 *
 * @note This function does **not** free the 'matrices' array itself-only the
 *       matrices it points to. Free the container separately if it was
 *       heap-allocated.
 */
static void matrix_data_array_free(double*** matrices, int num_of_matrices, int rows) {
    int i;
    for (i = 0; i < num_of_matrices; i++) {
        if(matrices[i]){
            matrix_data_free(matrices[i], rows);
        }
    }
}

/* ======================= Matrix File I/O ======================= */

/**
 * @brief Scan a numeric CSV-like file to determine matrix shape and validate consistency.
 *
 * Parses the stream as rows of doubles separated by commas (','), with each row
 * terminated by a newline ('\n'). While scanning, it verifies that every row
 * has the same number of values. On success, the function
 * rewinds the file and writes the inferred {rows, cols} into out_shape.
 *
 *
 * @param file      Open FILE* positioned at the beginning of the data to scan.
 *                  Must remain readable throughout the call.
 * @param out_shape Output pointer to receive the inferred shape {rows, cols}.
 *
 * @return STATUS_OK on success; otherwise STATUS_ERROR if:
 *         - file or out_shape is NULL,
 *         - a token cannot be parsed as a double,
 *         - an unexpected delimiter appears,
 *         - rows have inconsistent column counts,
 *         - no data (rows == 0 or cols == 0).
 */
static Status scan_and_validate_shape(FILE* file, Shape* out_shape) {
    int rows_cnt = 0, cols_cnt = -1, curr_cols_cnt = 0;
    double num;
    char ch;

    if (!file || !out_shape) {
        return STATUS_ERROR;
    }

    while (fscanf(file, "%lf%c", &num, &ch) == 2) {
        curr_cols_cnt++;
        if (ch == '\n') {
            if (cols_cnt != -1 && cols_cnt != curr_cols_cnt) {
                return STATUS_ERROR;
            }
            cols_cnt = curr_cols_cnt;
            curr_cols_cnt = 0;
            rows_cnt++;
        } else if (ch != '\n' && ch != ',') {
            return STATUS_ERROR;
        }
    }

    if (rows_cnt == 0 || cols_cnt == 0) {
        return STATUS_ERROR;
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
        return STATUS_ERROR;
    }

    out_shape->cols = cols_cnt;
    out_shape->rows = rows_cnt;
    return STATUS_OK;
}

/**
 * @brief Populate a pre-allocated row-major matrix from a file.
 *
 * Reads exactly shape.rows x shape.cols doubles from file and stores
 * them into data[r][c].
 *
 * On success, the function rewinds file to the beginning.
 *
 * @param file  Open, readable stream positioned at the start of the matrix data.
 * @param data  Pointer to an already-allocated table of row pointers
 *              ('double**') with at least shape.rows rows and
 *              shape.cols columns per row.
 * @param shape The matrix dimensions to read {rows, cols}.
 *
 * @return STATUS_OK on success; otherwise STATUS_ERROR if:
 *         - file or data is NULL,
 *         - parsing a value+delimiter fails (unexpected EOF or wrong format),
 *         - rewinding the stream with fseek fails.
 *
 * @note This routine assumes the layout was validated earlier.
 */
static Status fill_allocated_matrix(FILE* file, double** data, Shape shape) {
    int row = 0, col;
    double num;
    char ch;

    if (!file || !data) {
        return STATUS_ERROR;
    }

    for (row = 0; row < shape.rows; row++) {
        for (col = 0; col < shape.cols; col++) {
            if (fscanf(file, "%lf%c", &num, &ch) != 2) {
                return STATUS_ERROR;
            }
            data[row][col] = num;
        }
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
        return STATUS_ERROR;
    }

    return STATUS_OK;
}

/**
 * @brief Load a matrix from a CSV-like file into a newly allocated buffer.
 *
 * Infers the matrix dimensions by scanning the stream, allocates a row-major
 * 2D array of doubles of that size, fills it with the parsed values, and
 * writes the result into out_data.
 *
 * On success, out_data->data points to an allocated table of row pointers,
 * and out_data->shape contains {rows, cols}.
 *
 * @param file       Open, readable stream positioned at the start of the data.
 * @param out_data Output location for the allocated data pointer and shape.
 *
 * @return STATUS_OK on success; STATUS_ERROR on failure (invalid input,
 *         bad format, allocation failure, or read error).
 *
 * @note On failure, any temporary allocations are freed and out_data is
 *       not modified.
 */
static Status matrix_data_load_from_file(FILE* file, Matrix* out_data) {
    Shape shape;
    Status status_code;
    double** data;

    if (!file || !out_data) {
        return STATUS_ERROR;
    }

    status_code = scan_and_validate_shape(file, &shape);
    if (status_code != STATUS_OK) {
        return STATUS_ERROR;
    }

    data = matrix_data_alloc(shape.rows, shape.cols);
    if (data == NULL) {
        return STATUS_ERROR;
    }

    status_code = fill_allocated_matrix(file, data, shape);
    if (status_code != STATUS_OK) {
        matrix_data_free(data, shape.rows);
        return STATUS_ERROR;
    }

    out_data->data = data;
    out_data->shape.cols = shape.cols;
    out_data->shape.rows = shape.rows;

    return STATUS_OK;
}

/**
 * @brief Load a matrix from a file path into a newly allocated buffer.
 *
 * Opens the file at file_path for reading, delegates parsing and allocation to
 * matrix_data_load_from_file, closes the stream, and returns the resulting status.
 *
 * On success, out_data->data points to an allocated table of row pointers,
 * and out_data->shape contains {rows, cols}. On failure, out_data is not
 * modified.
 *
 * @param file_path path to a readable file.
 * @param out_data Output location for the allocated data pointer and shape.
 *
 * @return STATUS_OK on success; STATUS_ERROR on failure (invalid arguments,
 *         open failure, format/validation failure, allocation failure, or read error).
 */
static Status matrix_data_load_from_path(const char* file_path, Matrix* out_data) {
    FILE* file;
    Status status_code;

    if (!file_path || !out_data) {
        return STATUS_ERROR;
    }

    file = fopen(file_path, "r");
    if (!file) {
        return STATUS_ERROR;
    }

    status_code = matrix_data_load_from_file(file, out_data);
    fclose(file);

    return status_code;
}

/* ======================= Metrics (Distances & Norms) ======================== */

/**
 * @brief Computes the Euclidean distance between two vectors.
 *
 * @param vector1 Pointer to the first vector (length dim).
 * @param vector2 Pointer to the second vector (length dim).
 * @param dim     Number of elements in each vector (must be > 0).
 *
 * @return The Euclidean distance as a double.
 */
static double euclidean_distance(double* vector1, double* vector2, int dim) {
    int i;
    double sum = 0;
    
    for (i = 0; i < dim; i++) {
        sum += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);
    }

    return sqrt(sum);
}

/**
 * @brief Computes the squared Frobenius norm of a matrix.
 *
 * Adds up the square of every element in the matrix and returns the total.
 * The matrix is given as an array of row pointers and is not modified.
 *
 * @param data Array of row pointers to the matrix.
 * @param rows Number of rows (must be > 0).
 * @param cols Number of columns (must be > 0).
 *
 * @return The squared Frobenius norm on success; -1 if inputs are invalid.
 */
static double calculate_frobenius_norm_squared(double** data, int rows, int cols) {
    int i, j;
    double sum = 0;

    if (!data || rows <= 0 || cols <= 0) {
        return -1;
    }

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            sum += data[i][j] * data[i][j];
        }
    }

    return sum;
}


/* ======================= Matrix Algebra Utilities =========================== */
/**
 * @brief Sums the elements of a vector.
 *
 * Computes the sum of vector[i] for i = 0,1,...,dim-1.
 *
 * @param vector Pointer to the vector (length dim).
 * @param dim    Number of elements in the vector.
 *
 * @return The sum of the elements as a double.
 */
static double sum_vector_coordinates(double* vector, int dim) {
    int i;
    double sum = 0;

    for (i = 0; i < dim; i++) {
        sum += vector[i];
    }

    return sum;
}

/**
 * @brief Returns the transpose of a matrix.
 *
 * Allocates a new matrix of size cols*rows and sets out[j][i] = data[i][j]
 * for all valid indices. The input is treated as read-only.
 *
 * @param data Array of row pointers to the source matrix.
 * @param rows Number of rows in the source matrix (must be > 0).
 * @param cols Number of columns in the source matrix (must be > 0).
 *
 * @return A newly allocated cols*rows matrix containing the transpose on success;
 *         NULL on invalid input or allocation failure.
 */
static double** matrix_data_transpose(double** data, int rows, int cols) {
    int i, j;
    double** transposed_data;
    
    if (!data || rows <= 0 || cols <= 0) {
        return NULL;
    }

    transposed_data = matrix_data_alloc(cols, rows);
    if (!transposed_data) {
        return NULL;
    }

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            transposed_data[j][i] = data[i][j];
        }
    }

    return transposed_data;
}

/**
 * @brief Element-wise subtraction of two matrices.
 *
 * Allocates a new rows*cols matrix and sets result[i][j] =
 * data1[i][j] - data2[i][j] for all valid indices. The inputs are
 * treated as read-only.
 *
 * @param data1  Array of row pointers to the left-hand matrix.
 * @param data2  Array of row pointers to the right-hand matrix.
 * @param rows   Number of rows (must be > 0).
 * @param cols   Number of columns (must be > 0).
 *
 * @return A newly allocated rows*cols matrix containing the differences
 *         on success; NULL on invalid input or allocation failure.
 */
static double** matrix_data_subtract(double** data1, double** data2, int rows, int cols) {
    int i, j;
    double** diff_data;

    if (!data1 || !data2 || rows <= 0 || cols <= 0) {
        return NULL;
    }

    diff_data = matrix_data_alloc(rows, cols);
    if (!diff_data) {
        return NULL;
    }

    for(i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            diff_data[i][j] = data1[i][j] - data2[i][j];
        }
    }

    return diff_data;
}

/**
 * @brief Multiply two matrices (C = A * B) and return a newly allocated result.
 *
 * Computes the standard matrix product where A is rows1 x cols1 and
 * B is rows2 x cols2. The inputs are treated as read-only. The function
 * validates that cols1 == rows2 and that all dimensions are positive,
 * allocates a rows1 x cols2 result, and accumulates the dot products.
 *
 * @param data1  Left operand (A), array of row pointers of size rows1.
 * @param data2  Right operand (B), array of row pointers of size rows2.
 * @param rows1  Number of rows in A (must be > 0).
 * @param cols1  Number of columns in A (and rows in B) (must be > 0).
 * @param rows2  Number of rows in B (must be > 0).
 * @param cols2  Number of columns in B (must be > 0).
 *
 * @return A newly allocated rows1 x cols2 matrix on success;
 *         NULL if inputs are invalid, dimensions are incompatible,
 *         or allocation fails.
 */
static double** matrix_data_multiply(double** data1, double** data2,
    int rows1, int cols1, int rows2, int cols2) {
        int i, j, k;
        double** multiplied_data;

        if (!data1 || !data2 || rows1 <= 0 || cols1 <= 0 || rows2 <= 0 || cols2 <= 0 || cols1 != rows2) {
            return NULL;
        }

        multiplied_data = matrix_data_alloc(rows1, cols2);
        if (!multiplied_data) {
            return NULL;
        }

        for(i = 0; i < rows1; i++) {
            for (j = 0; j < cols2; j++) {
                for (k = 0; k < cols1; k++) {
                    multiplied_data[i][j] += data1[i][k]*data2[k][j];
                }
            }
        }

        return multiplied_data;
}

/**
 * @brief Multiply three matrices in sequence: (A * B) * C.
 *
 * Computes the product of three matrices by first forming an intermediate
 * AB = A * B and then multiplying by C to get ABC. The inputs are treated
 * as read-only. The intermediate AB is freed before returning.
 *
 * Dimensions:
 *   - A is rows1 x cols1
 *   - B is rows2 x cols2
 *   - C is rows3 x cols3
 *   - Requires cols1 == rows2 and cols2 == rows3
 *   - Result is rows1 x cols3
 *
 * Ownership: the caller must free the returned matrix with
 *   matrix_data_free(result, rows1)
 * when it is no longer needed.
 *
 * @param data1  Left operand A.
 * @param data2  Middle operand B.
 * @param data3  Right operand C.
 * @param rows1  Rows in A.
 * @param cols1  Columns in A.
 * @param rows2  Rows in B.
 * @param cols2  Columns in B.
 * @param rows3  Rows in C.
 * @param cols3  Columns in C.
 *
 * @return A newly allocated rows1 x cols3 matrix on success;
 *         NULL if any multiplication fails or dimensions are incompatible.
 */
static double** matrix_data_multiply_three(double** data1, double** data2, double** data3,
    int rows1, int cols1, int rows2, int cols2, int rows3, int cols3) {
        double** data1data2;
        double** all_three_multiplied;

        data1data2 = matrix_data_multiply(data1, data2, rows1, cols1, rows2, cols2);
        if (!data1data2) {
            return NULL;
        }

        all_three_multiplied = matrix_data_multiply(data1data2, data3, rows1, cols2, rows3, cols3);
        matrix_data_free(data1data2, rows1);
        if (!all_three_multiplied) {
            return NULL;
        }

        return all_three_multiplied;
}

/**
 * @brief Compute the element-wise inverse square root of an n*n matrix.
 *
 * Allocates a new matrix of the same size and sets each entry to:
 *   - 1 / sqrt(data[i][j]) if data[i][j] > 0
 *   - 0 if data[i][j] == 0  (avoid division by zero)
 *
 * If any value is negative, the function frees anything it already
 * allocated and returns NULL.
 *
 * @param data Input matrix (n*n), treated as read-only.
 * @param n    Number of rows/columns (must be > 0).
 *
 * @return A newly allocated n*n matrix on success, or NULL if input
 *         is invalid, memory allocation fails, or a negative entry
 *         is found. Caller must free with matrix_data_free(..., n).
 */
static double** matrix_data_elementwise_inv_sqrt(double** data, int n) {
    int i, j;
    double** inv_sqrt_data;

    if (!data || n <= 0) {
        return NULL;
    }

    inv_sqrt_data = matrix_data_alloc(n, n);
    if (!inv_sqrt_data) {
        return NULL;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (data[i][j] < 0) {
                matrix_data_free(inv_sqrt_data, n);
                return NULL;
            } else if (data[i][j] == 0) {
                inv_sqrt_data[i][j] = 0; /* Define 1/sqrt(0) as 0 to avoid division by zero */
            } else {
            inv_sqrt_data[i][j] = 1.0 / sqrt(data[i][j]);
            }
        }
    }

    return inv_sqrt_data;
}

/* ======================= Similarity Matrix Utilities ========================= */

/**
 * @brief Computes the similarity between two vectors.
 *
 * Uses the Euclidean distance d between vector1 and vector2 and returns
 *   exp(-BETA * d^2).
 *
 * @param vector1 Pointer to the first vector (length dim).
 * @param vector2 Pointer to the second vector (length dim).
 * @param dim     Number of elements in each vector.
 *
 * @return Similarity between two vectors.
 */
static double similarity_of_two_vectors(double* vector1, double* vector2, int dim) {
    double distance;
    distance = euclidean_distance(vector1, vector2, dim);
    return exp(-BETA * distance * distance);
}

/**
 * @brief Build an n*n similarity matrix from row vectors.
 *
 * For each pair of rows i,j in 'data', computes the similarity
 *   exp(-BETA * ||data[i] - data[j]||^2)
 * using Euclidean distance. The diagonal is set to 0.0.
 *
 * @param sim_matrix    Out: receives the allocated n*n similarity matrix.
 * @param data          Input data as an array of 'n' row vectors,
 *                      each of length 'dim'. Treated as read-only.
 * @param n Number of rows/cols in the similarity matrix.
 * @param dim           Length of each row vector in 'data'.
 */
void compute_similarity_matrix(double*** sim_matrix, double** data, int n, int dim) {
    int i, j;
    
    *sim_matrix = NULL;
    *sim_matrix = matrix_data_alloc(n, n);
    if (!*sim_matrix) {
        return;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                (*sim_matrix)[i][j] = 0.0;
            } else {
            (*sim_matrix)[i][j] = similarity_of_two_vectors(data[i], data[j], dim);
            }
        }
    }
}

/**
 * @brief Build a diagonal degree matrix from a similarity matrix.
 *
 * For each row i of sim_matrix, sets D[i][i] = sum(sim_matrix[i][j] for j=0,1,...,n-1).
 * Off-diagonal entries remain 0.
 *
 * @param degree_matrix  Out: address of a (double**) to receive the new matrix.
 * @param sim_matrix     In: n*n similarity matrix (read-only).
 * @param n              Dimension (rows/cols).
 */
void compute_diagonal_degree_matrix (double*** degree_matrix, double** sim_matrix, int n) {
    int i;
    
    *degree_matrix = matrix_data_alloc(n, n);
    if (!*degree_matrix) {
        return;
    }

    for (i = 0; i < n; i++) {
        (*degree_matrix)[i][i] = sum_vector_coordinates(sim_matrix[i], n);
    }
}

/**
 * @brief Compute the normalized similarity matrix
 *        W = D^{-1/2} A D^{-1/2}.
 *
 * Here A is the similarity matrix and D is the diagonal degree
 * matrix with D[i,i] = sum_j A[i,j]. We build D^{-1/2} by taking the inverse
 * square root of the diagonal (leaving off-diagonals zero) and then compute
 * the sandwich product above.
 *
 * Inputs are read-only. On success, *out_norm_sim_matrix receives a newly
 * allocated n*n matrix W. Caller frees it with matrix_data_free(*out_norm_sim_matrix, n).
 * If D has a negative diagonal entry or any allocation/multiply fails,
 * *out_norm_sim_matrix is set to NULL.
 *
 * @param out_norm_sim_matrix  OUT: normalized matrix W.
 * @param sim_matrix           IN : A (n*n).
 * @param degree_matrix        IN : D (n*n, diagonal).
 * @param n                    Dimension (n > 0).
 */
void compute_normalized_similarity_matrix(double*** out_norm_sim_matrix, double** sim_matrix,
    double** degree_matrix, int n) {
        double** D_inv_sqrt;
        double** norm_sim_matrix;

        D_inv_sqrt = matrix_data_elementwise_inv_sqrt(degree_matrix, n);
        if (!D_inv_sqrt) {
            *out_norm_sim_matrix = NULL;
            return;
        }

        norm_sim_matrix = matrix_data_multiply_three(D_inv_sqrt, sim_matrix, D_inv_sqrt, n, n, n, n, n, n);
        matrix_data_free(D_inv_sqrt, n);
        if (!norm_sim_matrix) {
            *out_norm_sim_matrix = NULL;
            return;
        }

        *out_norm_sim_matrix = norm_sim_matrix;
        return;
}

/* ======================= Output / Printing Utilities ======================== */

/**
 * @brief Print a rows*cols matrix to stdout in CSV format.
 *
 * Emits exactly 'rows' lines; each line contains 'cols' values formatted
 * with "%.4f" and separated by commas (no trailing comma). A newline is
 * printed after each row.
 *
 * @param data Pointer to an array of 'rows' row pointers (read-only).
 * @param rows Number of rows to print (> 0).
 * @param cols Number of columns to print (> 0).
 */
static void matrix_data_print(double** data, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f", data[i][j]);
            if (j < cols - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

/**
 * @brief Print a square similarity matrix.
 *
 * Verifies that sim_matrix is non-NULL. If validation fails, returns STATUS_ERROR. Otherwise,
 * prints the matrix to stdout in the same CSV format as matrix_data_print
 * (comma-separated, "%.4f", one row per line) and returns STATUS_OK.
 *
 * @param matrix     Matrix carrying the shape (rows, cols).
 * @param sim_matrix Pointer to an n*n similarity matrix (read-only).
 *
 * @return STATUS_OK on success; STATUS_ERROR on invalid input.
 */
static Status similarity_matrix_print(Matrix matrix, double** sim_matrix) {
    if (!sim_matrix) {
        return STATUS_ERROR;
    }

    matrix_data_print((double**)sim_matrix, matrix.shape.rows, matrix.shape.rows);
    return STATUS_OK;
}


/**
 * @brief Print the diagonal degree matrix built from a similarity matrix.
 *
 * Checks that sim_matrix is non-NULL. Constructs the degree matrix D where D[i][i] is the
 * row sum of sim_matrix[i][*], prints D to stdout in the same CSV format
 * as matrix_data_print (comma-separated, "%.4f", one row per line),
 * then frees D.
 *
 * On failure (invalid input or allocation error), nothing is printed and
 * STATUS_ERROR is returned.
 *
 * @param matrix     Matrix carrying the shape (rows, cols).
 * @param sim_matrix Pointer to an n*n similarity matrix (read-only).
 *
 * @return STATUS_OK on success; STATUS_ERROR on invalid input or allocation failure.
 */

static Status diagonal_degree_matrix_print(Matrix matrix, double** sim_matrix) {
    double** degree_matrix;

    if (!sim_matrix) {
        return STATUS_ERROR;
    }

    compute_diagonal_degree_matrix(&degree_matrix, sim_matrix, matrix.shape.rows);
    if (!degree_matrix) {
        return STATUS_ERROR;
    }

    matrix_data_print((double**)degree_matrix, matrix.shape.rows, matrix.shape.rows);
    matrix_data_free(degree_matrix, matrix.shape.rows);
    return STATUS_OK;
}

/**
 * @brief Print the normalized similarity matrix W = D^{-1/2} A D^{-1/2}.
 *
 * Checks that sim_matrix is non-NULL. Constructs the diagonal degree matrix D from sim_matrix,
 * computes W, prints W to stdout in the same CSV format as matrix_data_print
 * (comma-separated, "%.4f", one row per line), then frees D and W.
 *
 * On failure (invalid input, allocation error, or negative diagonal entry),
 * nothing is printed and STATUS_ERROR is returned.
 *
 * @param matrix     Matrix carrying the shape (rows, cols).
 * @param sim_matrix Pointer to an n*n similarity matrix (read-only).
 *
 * @return STATUS_OK on success; STATUS_ERROR on invalid input, allocation failure,
 *         or negative diagonal entry in D.
 */
static Status normalized_similarity_matrix_print(Matrix matrix, double** sim_matrix) {
    double** degree_matrix;
    double** norm_sim_matrix;
    
    if (!sim_matrix) {
        return STATUS_ERROR;
    }

    compute_diagonal_degree_matrix(&degree_matrix, sim_matrix, matrix.shape.rows);
    if (!degree_matrix) {
        return STATUS_ERROR;
    }

    compute_normalized_similarity_matrix(&norm_sim_matrix, sim_matrix, degree_matrix, matrix.shape.rows);
    matrix_data_free(degree_matrix, matrix.shape.rows);
    if (!norm_sim_matrix) {
        return STATUS_ERROR;
    }

    matrix_data_print((double**)norm_sim_matrix, matrix.shape.rows, matrix.shape.rows);
    matrix_data_free(norm_sim_matrix, matrix.shape.rows);
    return STATUS_OK;
}

/* ======================= SymNMF Optimization ================================ */

/**
 * @brief Update a single entry H[i][j] using the SymNMF update rule.
 *
 * Uses the formula:
 *   H[i][j] = H[i][j] * (1 - BETA + BETA * (WH[i][j] / HHtH[i][j]))
 * If HHtH[i][j] == 0, sets H[i][j] = 0 to avoid division by zero.
 *
 * @param next_H  Out: table of row pointers to receive the updated value.
 * @param H      In: current factor matrix (read-only).
 * @param WH     In: precomputed W*H matrix (read-only).
 * @param HHtH   In: precomputed H*H^T*H matrix (read-only).
 * @param i      Row index of the entry to update (0 <= i < rows).
 * @param j      Column index of the entry to update (0 <= j < cols).
 */
static void symnmf_update_H_entry(double** next_H, double** H, double** WH, double** HHtH, int i, int j) {
    double update_factor;
    if (HHtH[i][j] == 0) {
        next_H[i][j] = 0;
    } else {
        update_factor = 1 - BETA + (BETA * (WH[i][j] / HHtH[i][j]));
        next_H[i][j] = H[i][j] * update_factor;
    }
}

/**
 * @brief Update all entries of H using the SymNMF update rule.
 *
 * Iterates over every entry H[i][j] and applies symnmf_update_H_entry
 * to compute the new value, storing it in next_H[i][j].
 *
 * @param next_H  Out: table of row pointers to receive the updated matrix.
 * @param H      In: current factor matrix (read-only).
 * @param WH     In: precomputed W*H matrix (read-only).
 * @param HHtH   In: precomputed H*H^T*H matrix (read-only).
 * @param rows   Number of rows in H (must be > 0).
 * @param cols   Number of columns in H (must be > 0).
 */
static void symnmf_fill_next_H(double** next_H, double** H, double** WH, double** HHtH, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            symnmf_update_H_entry(next_H, H, WH, HHtH, i, j);
        }
    }
}

/**
 * @brief Replace the current H matrix with the next_H matrix.
 *
 * Frees the memory occupied by H and sets H to point to next_H.
 * Sets next_H to NULL to avoid dangling pointers.
 *
 * @param H        In/Out: address of the current H pointer (rows x k).
 * @param next_H   In/Out: address of the next_H pointer (rows x k).
 * @param rows     Number of rows in H and next_H (must be > 0).
 */
static void set_H_to_next(double*** H, double*** next_H, int rows) {
    matrix_data_free(*H, rows);
    *H = *next_H;
    *next_H = NULL;
}

/**
 * @brief Compute the intermediate matrices WH and HHtH for updating H.
 *
 * Allocates and computes:
 *   - WH = W * H
 *   - HHtH = H * H^T * H
 * The caller must free both matrices with matrix_data_free(..., rows).
 *
 * @param H        In: current factor matrix (read-only).
 * @param W        In: normalized similarity matrix (read-only).
 * @param matrices Out: array of two (double**) pointers to receive WH and HHtH.
 *                 Must have space for at least 2 entries.
 * @param rows     Number of rows in H and W (must be > 0).
 * @param cols     Number of columns in H (must be > 0).
 *
 * @return STATUS_OK on success; STATUS_ERROR if any allocation or multiplication fails.
 */
static Status symnmf_compute_H_update_intermediate_matrices(double** H, double** W, double*** matrices,
    int rows, int cols) {
        int i;
        double** Ht = NULL;

        for (i = 0; i < 3; i++) {
            matrices[i] = NULL;
        }

        Ht = matrix_data_transpose(H, rows, cols);
        if (!Ht) {
            return STATUS_ERROR;
        }

        /* WH */
        matrices[0] = matrix_data_multiply(W, H, rows, rows, rows, cols);
        if (!matrices[0]) {
            matrix_data_free(Ht, cols);
            return STATUS_ERROR;
        }

        /* HHtH */
        matrices[1] = matrix_data_multiply_three(H, Ht, H, rows, cols, cols, rows, rows, cols);
        matrix_data_free(Ht, cols);

        if (!matrices[1]) {
            matrix_data_array_free(matrices, 1, rows); /* frees matrices[0] */
            return STATUS_ERROR;
        }

        return STATUS_OK;
}

/**
 * @brief Perform a single SymNMF update step on H.
 *
 * Workflow:
 *   1) Allocate next_H (rows x k).
 *   2) Compute intermediates into matrices[]:
 *        matrices[0] = W * H        // WH
 *        matrices[1] = H * H^T * H  // HHtH
 *   3) Fill next_H using the SymNMF update rule.
 *   4) matrices[2] = next_H - H      // diff
 *   5) Set *has_converged = (||diff||_F^2 < EPSILON).
 *   6) Replace *H with next_H in all cases (converged or not).
 *   7) Free matrices[0], matrices[1], matrices[2].
 *
 * Ownership and errors:
 *   - On success: the old *H is freed, *H now points to next_H, and *out_next_H is set to NULL.
 *   - On failure: frees any temporaries, leaves *H unchanged, sets *out_next_H = NULL,
 *     and returns STATUS_ERROR.
 *
 * @param H              In/Out: address of the current H pointer (rows x k).
 * @param out_next_H      Out: receives the newly allocated H before it is transferred to *H.
 *                       Set to NULL on failure. Consumed on success.
 * @param W              In: normalized similarity matrix (rows x rows), read-only.
 * @param matrices       Scratch array of length 3 used as:
 *                       [0]=WH, [1]=HHtH, [2]=diff. Freed inside on success or failure.
 * @param rows           Number of rows in H and W (> 0).
 * @param k              Number of columns in H (> 0).
 * @param EPSILON        Convergence threshold applied to ||diff||_F^2 (> 0).
 * @param has_converged  Out: set to 1 if ||diff||_F^2 < EPSILON; otherwise 0.
 *
 * @return STATUS_OK on success; STATUS_ERROR on allocation or compute failure.
 */
static Status symnmf_perform_H_step(double*** H, double*** out_next_H, double** W, double** matrices[3], const int rows, const int k, const double EPSILON, int* has_converged) {
    *out_next_H = matrix_data_alloc(rows, k);
    if (!*out_next_H) {
        return STATUS_ERROR;
    }
    
    if (symnmf_compute_H_update_intermediate_matrices(*H, W, matrices, rows, k) != STATUS_OK) {
        matrix_data_free(*out_next_H, rows);
        *out_next_H = NULL;
        return STATUS_ERROR;

    }

    symnmf_fill_next_H(*out_next_H, *H, matrices[0], matrices[1], rows, k);
    matrices[2] = matrix_data_subtract(*out_next_H, *H, rows, k);
    if (!matrices[2]) {
        matrix_data_array_free(matrices, 2, rows);
        matrix_data_free(*out_next_H, rows);
        return STATUS_ERROR;
    }

    *has_converged = (calculate_frobenius_norm_squared((double**)matrices[2], rows, k) < EPSILON);

    set_H_to_next(H, out_next_H, rows);
    matrix_data_array_free(matrices, 3, rows);
    return STATUS_OK;
}

/**
 * @brief Iteratively update H until convergence or max iterations reached.
 *
 * Repeatedly applies symnmf_perform_H_step to update H until either
 * convergence is detected (||next_H - H||_F^2 < EPSILON) or the maximum
 * number of iterations MAX_ITER is reached.
 *
 * On success, *H points to the final factor matrix. On failure, *H is
 * freed and set to NULL.
 *
 * @param H         In/Out: address of the current H pointer (rows x k).
 *                  Updated in place to point to the final matrix on success.
 *                  Set to NULL on failure.
 * @param W         In: normalized similarity matrix (rows x rows), read-only.
 * @param rows      Number of rows in H and W (> 0).
 * @param k         Number of columns in H (> 0).
 * @param MAX_ITER  Maximum number of iterations to perform (> 0).
 * @param EPSILON   Convergence threshold applied to ||next_H - H||_F^2 (> 0).
 */
void symnmf_iterate_H_until_convergence(double*** H, double** W, int rows, int k, const int MAX_ITER, const double EPSILON) {
    double** next_H =NULL;
    double** matrices[3] = {NULL, NULL, NULL};
    int iter = 0, has_converged = 0;
    Status status_code;
    if(!H || !*H || !W || rows <= 0 || k <= 0 || MAX_ITER <= 0 || EPSILON <= 0) {
        return;
    }

    do {
        status_code = symnmf_perform_H_step(H, &next_H, W, matrices, rows, k, EPSILON, &has_converged);
    } while (!has_converged && ++iter < MAX_ITER && status_code == STATUS_OK);

    if (status_code != STATUS_OK) {
        if (next_H) {
            matrix_data_free(next_H, rows);
        }

        if (matrices[0] || matrices[1] || matrices[2]) {
            matrix_data_array_free(matrices, 3, rows);
        }

        if (*H) {
            matrix_data_free(*H, rows);
            *H = NULL;
        }
    }
}

/* ======================= Command-Line Interface ============================= */
typedef struct {
    const char* filename;
    const char* goal;
} Arguments;

/* Indices into argv and expected argc for the CLI:
 * argv[SYM_ARG_PROG]     = program name
 * argv[SYM_ARG_GOAL]     = goal string: "sym" | "ddg" | "norm"
 * argv[SYM_ARG_FILENAME] = path to input data file
 * SYM_ARGC_EXPECTED      = expected argc (3) */
enum { SYM_ARG_PROG = 0,
     SYM_ARG_GOAL = 1,
     SYM_ARG_FILENAME = 2,
     SYM_ARGC_EXPECTED = 3 };

/**
 * @brief Parse command-line arguments into an Arguments struct.
 *
 * Expects exactly three argv entries (program name, goal, filename).
 * Validates pointers and argc, then assigns:
 *   out_args->goal     = argv[SYM_ARG_GOAL];
 *   out_args->filename = argv[SYM_ARG_FILENAME];
 *
 * On any error (NULL pointers, wrong argc, or NULL fields), returns STATUS_ERROR
 * and does not modify out_args.
 *
 * @param argc       Argument count received by main.
 * @param argv       Argument vector received by main.
 * @param out_args   Output: populated with goal and filename on success.
 *
 * @return STATUS_OK on success; STATUS_ERROR on invalid input or format.
 */
static Status parse_arguments(int argc, char** argv, Arguments* out_args) {
    if((!out_args) || (argc != SYM_ARGC_EXPECTED) || (!argv)) {
        return STATUS_ERROR;
    }

    out_args->goal = argv[SYM_ARG_GOAL];
    out_args->filename = argv[SYM_ARG_FILENAME];

    if ((out_args->goal == NULL) || (out_args->filename == NULL)) {
        return STATUS_ERROR;
    }

    return STATUS_OK;
}

/* Goal callback type: prints/handles a matrix derived from sim_matrix. */
typedef Status sym_goal_fn(Matrix input_matrix, double **sim_matrix);
typedef sym_goal_fn* sym_goal_fn_ptr;

/**
 * @brief Map a goal name string to the corresponding function pointer.
 *
 * Recognizes the following goal names:
 *   - "norm" -> &normalized_similarity_matrix_print
 *   - "ddg"  -> &diagonal_degree_matrix_print
 *   - "sym"  -> &similarity_matrix_print
 *
 * If goal_name is NULL, out_goal_fn is NULL, or the goal name is unrecognized,
 * returns STATUS_ERROR and does not modify out_goal_fn. Otherwise, sets
 * *out_goal_fn to the corresponding function pointer and returns STATUS_OK.
 *
 * @param goal_name     Input goal name string (e.g., "norm", "ddg", "sym").
 * @param out_goal_fn   Output: receives the corresponding function pointer on success.
 *
 * @return STATUS_OK on success; STATUS_ERROR on invalid input or unrecognized goal name.
 */
static Status get_goal_function(const char* goal_name, sym_goal_fn_ptr* out_goal_fn) {
    if (!goal_name || !out_goal_fn) {
        return STATUS_ERROR;
    }

    if (strcmp(goal_name, "norm") == 0) {
        *out_goal_fn = &normalized_similarity_matrix_print;
    } else if (strcmp(goal_name, "ddg") == 0) {
        *out_goal_fn = &diagonal_degree_matrix_print;
    } else if (strcmp(goal_name, "sym") == 0) {
        *out_goal_fn = &similarity_matrix_print;
    } else {
        return STATUS_ERROR;
    }

    return STATUS_OK;
}

/**
 * @brief Compute the similarity matrix and invoke the specified goal function.
 *
 * Given an input_matrix and a goal function pointer, computes the similarity
 * matrix and then calls the goal function with (input_matrix, sim_matrix).
 * Frees the similarity matrix before returning.
 *
 * If goal_fn is NULL or any step fails, returns STATUS_ERROR.
 *
 * @param input_matrix  Input data matrix (read-only).
 * @param goal_fn       Function pointer to the goal function to invoke.
 *
 * @return The status code returned by the goal function on success;
 *         STATUS_ERROR on invalid input or failure.
 */
static Status compute_similarity_matrix_and_run_goal(Matrix input_matrix, sym_goal_fn_ptr goal_fn) {
    Status status_code;
    double** sim_matrix;

    if (!goal_fn) {
        return STATUS_ERROR;
    }

    compute_similarity_matrix(&sim_matrix, input_matrix.data, input_matrix.shape.rows, input_matrix.shape.cols);
    if (!sim_matrix) {
        return STATUS_ERROR;
    }

    status_code = goal_fn(input_matrix, sim_matrix);
    matrix_data_free(sim_matrix, input_matrix.shape.rows);
    
    return status_code;
}

/**
 * @brief Load data from a file, compute similarity matrix, and run the goal function.
 * 
 * Loads a matrix from the specified file path, computes its similarity matrix,
 * and invokes the provided goal function with the loaded matrix and the computed
 * similarity matrix. Frees all allocated resources before returning.
 * 
 * @param filepath  Path to the input data file.
 * @param goal_fn   Function pointer to the goal function to invoke.
 * 
 * @return STATUS_OK if all operations succeed; STATUS_ERROR on any failure.
 * 
 */
static Status load_data_and_run_goal(const char *filepath, sym_goal_fn_ptr goal_fn) {
    Matrix input_matrix;
    Status status_code;

    if (!filepath || !goal_fn) {
        return STATUS_ERROR;
    }

    status_code = matrix_data_load_from_path(filepath, &input_matrix);
    if (status_code != STATUS_OK) {
        return STATUS_ERROR;
    }

    status_code = compute_similarity_matrix_and_run_goal(input_matrix, goal_fn);
    matrix_data_free(input_matrix.data, input_matrix.shape.rows);

    return status_code;
}

/**
 * Entry point: parse args, resolve goal, run, and report errors.
 *
 * Usage:
 *   prog <goal> <filename>
 *     goal     one of: "sym", "ddg", "norm"
 *     filename path to CSV-like data file
 *
 * Behavior:
 *   - Parses CLI arguments.
 *   - Maps <goal> to the corresponding print routine.
 *   - Loads data, builds similarity matrix, and prints the requested output.
 *
 * Return codes:
 *   0  on success
 *   1  on failure (via ASSERT_OK macro)
 *
 * Side effects:
 *   Prints matrices to stdout; prints a generic error message on failure.
 */
int main(int argc, char** argv) {
    Arguments args;
    Status status_code;
    sym_goal_fn_ptr goal_fn;

    status_code = parse_arguments(argc, argv, &args);
    ASSERT_OK(status_code);

    status_code = get_goal_function(args.goal, &goal_fn);
    ASSERT_OK(status_code);

    status_code = load_data_and_run_goal(args.filename, goal_fn);
    ASSERT_OK(status_code);

    return 0;
} 