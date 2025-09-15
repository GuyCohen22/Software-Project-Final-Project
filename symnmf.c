#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const double BETA = 0.5;

typedef enum { 
    STATUS_OK = 0,
    STATUS_ERROR = 1,
} Status;

typedef struct {
    int rows;
    int cols;
} Shape;

typedef struct {
    double** data;
    Shape shape;
} Matrix;

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
 * @note This function does **not** free the 'matrices' array itself—only the
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
 * writes the result into out_matrix.
 *
 * On success, out_matrix->data points to an allocated table of row pointers,
 * and out_matrix->shape contains {rows, cols}.
 *
 * @param file       Open, readable stream positioned at the start of the data.
 * @param out_matrix Output location for the allocated data pointer and shape.
 *
 * @return STATUS_OK on success; STATUS_ERROR on failure (invalid input,
 *         bad format, allocation failure, or read error).
 *
 * @note On failure, any temporary allocations are freed and out_matrix is
 *       not modified.
 */
static Status matrix_load_from_file(FILE* file, Matrix* out_matrix) {
    Shape shape;
    Status status_code;
    double** data;

    if (!file || !out_matrix) {
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

    out_matrix->data = data;
    out_matrix->shape.cols = shape.cols;
    out_matrix->shape.rows = shape.rows;

    return STATUS_OK;
}

/**
 * @brief Load a matrix from a file path into a newly allocated buffer.
 *
 * Opens the file at file_path for reading, delegates parsing and allocation to
 * matrix_load_from_file, closes the stream, and returns the resulting status.
 *
 * On success, out_matrix->data points to an allocated table of row pointers,
 * and out_matrix->shape contains {rows, cols}. On failure, out_matrix is not
 * modified.
 *
 * @param file_path path to a readable file.
 * @param out_matrix Output location for the allocated data pointer and shape.
 *
 * @return STATUS_OK on success; STATUS_ERROR on failure (invalid arguments,
 *         open failure, format/validation failure, allocation failure, or read error).
 */
static Status matrix_load_from_path(const char* file_path, Matrix* out_matrix) {
    FILE* file;
    Status status_code;

    if (!file_path || !out_matrix) {
        return STATUS_ERROR;
    }

    file = fopen(file_path, "r");
    if (!file) {
        return STATUS_ERROR;
    }

    status_code = matrix_load_from_file(file, out_matrix);
    fclose(file);

    return status_code;
}
