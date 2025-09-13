#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const double BETA = 0.5;

typedef enum { 
    SUCCESS = 0,
     GENERAL_ERROR = 1 
} sym_status;

typedef struct {
    int rows;
    int cols;
} shape;

typedef struct {
    double** data;
    shape    size;
} matrix;

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





