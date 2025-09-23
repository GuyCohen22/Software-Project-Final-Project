#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include "symnmf.h"

const int MAX_ITER = 300;

/**
 * @brief Convert a Python list of lists (2D array) to a C-style 2D array (double**).
 *
 * Allocates a new double** array with the same dimensions as the input list of lists.
 * Each inner list must contain only float-convertible elements.
 *
 * @param list       Input Python list of lists (2D array).
 * @param out_rows   Output parameter to receive the number of rows in the resulting matrix.
 * @param out_cols   Output parameter to receive the number of columns in the resulting matrix.
 *
 * @return Pointer to the newly allocated double** array on success; NULL on failure.
 *         On failure, sets a Python exception (e.g., MemoryError).
 */
double** pylist_to_c_matrix(PyObject* list, int* out_rows, int* out_cols) {
    int rows, cols, i, j;
    double** matrix = NULL;
    PyObject *row;

    rows = (int)PyList_Size(list);
    if (rows == 0) {
        return NULL;
    }

    row = PyList_GetItem(list, 0);
    cols = (int)PyList_Size(row);
    if (cols == 0) {
        return NULL;
    }

    matrix = (double**)malloc(rows * sizeof(double*));
    if (!matrix) {
        PyErr_NoMemory();
        return NULL;
    }

    for (i = 0; i < rows; i++) {
        row = PyList_GetItem(list, i);
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (!matrix[i]) {
            matrix_data_free(matrix, i);
            PyErr_NoMemory();
            return NULL;
        }

        for (j = 0; j < cols; j++) {
            matrix[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
    *out_rows = rows;
    *out_cols = cols;
    return matrix;
}

/**
 * @brief Convert a C-style 2D array (double**) to a Python list of lists (2D array).
 *
 * Allocates a new Python list of lists with the same dimensions as the input matrix.
 *
 * @param matrix Input C-style 2D array (double**).
 * @param rows   Number of rows in the input matrix.
 * @param cols   Number of columns in the input matrix.
 *
 * @return New reference to a Python list of lists on success; NULL on failure.
 *         On failure, sets a Python exception (e.g., MemoryError).
 */
PyObject* c_matrix_to_pylist(double** matrix, int rows, int cols) {
    int i, j;
    PyObject *list, *row, *value;

    list = PyList_New(rows);
    if (!list) {
        PyErr_NoMemory();
        return NULL;
    }

    for (i = 0; i < rows; i++) {
        row = PyList_New(cols);
        if (!row) {
            Py_DECREF(list);
            PyErr_NoMemory();
            return NULL;
        }

        for (j = 0; j < cols; j++) {
            value = PyFloat_FromDouble(matrix[i][j]);
            if (!value) {
                Py_DECREF(row);
                Py_DECREF(list);
                PyErr_NoMemory();
                return NULL;
            }
            PyList_SetItem(row, j, value);
        }
        PyList_SetItem(list, i, row);
    }

    return list;
}

/**
 * @brief Python wrapper for compute_similarity_matrix (A = exp(-beta * ||xi - xj||^2).
 *
 * Expects a single argument: a Python list of lists representing a real-valued
 * data matrix X of shape (rows * cols). Each inner list is treated as a row
 * vector. The function converts X to a C matrix, computes the similarity
 * matrix A, and returns A as a Python list of lists with shape (rows * rows).
 *
 * @param self  Unused.
 * @param args  Tuple containing one object: data_list (list[list[float]]).
 *
 * @return New reference to a Python list of lists on success; NULL on failure.
 *         On failure, a Python exception is set (e.g., TypeError on bad args,
 *         MemoryError on allocation failure).
 */
static PyObject* py_compute_sym(PyObject* self, PyObject* args) {
    PyObject* data_list;
    PyObject* result_list;
    double** data;
    double** sim_matrix;
    int rows, cols;

    if (!PyArg_ParseTuple(args, "O", &data_list)) {
        return NULL;
    }

    data = pylist_to_c_matrix(data_list, &rows, &cols);
    if (!data) {
        return PyErr_NoMemory();
    }

    compute_similarity_matrix(&sim_matrix, data, rows, cols);
    if (!sim_matrix) {
        matrix_data_free(data, rows);
        return PyErr_NoMemory();
    }

    result_list = c_matrix_to_pylist(sim_matrix, rows, rows);
    matrix_data_free(data, rows);
    matrix_data_free(sim_matrix, rows);
    if (!result_list) {
        return PyErr_NoMemory();
    }
    
    return result_list;
}

/**
 * @brief Python wrapper for compute_diagonal_degree_matrix (D[i,i] = sum_j A[i,j]).
 *
 * Expects a single argument: a Python list of lists representing a square
 * similarity matrix A of shape (n * n). The function converts A to a C matrix,
 * computes the diagonal degree matrix D (zeros off-diagonal), and returns D as
 * a Python list of lists with shape (n * n).
 *
 * @param self  Unused.
 * @param args  Tuple containing one object: sim_matrix_list (list[list[float]]).
 *
 * @return New reference to a Python list of lists on success; NULL on failure.
 *         On failure, a Python exception is set (e.g., TypeError on bad args,
 *         MemoryError on allocation failure).
 */
static PyObject* py_compute_ddg(PyObject* self, PyObject* args) {
    PyObject* sim_matrix_list;
    PyObject* result_list;
    double** sim_matrix;
    double** degree_matrix;
    int n;

    if (!PyArg_ParseTuple(args, "O", &sim_matrix_list)) {
        return NULL;
    }

    sim_matrix = pylist_to_c_matrix(sim_matrix_list, &n, &n);
    if (!sim_matrix) {
        return PyErr_NoMemory();
    }

    compute_diagonal_degree_matrix(&degree_matrix, sim_matrix, n);
    if (!degree_matrix) {
        matrix_data_free(sim_matrix, n);
        return PyErr_NoMemory();
    }

    result_list = c_matrix_to_pylist(degree_matrix, n, n);
    matrix_data_free(sim_matrix, n);
    matrix_data_free(degree_matrix, n);
    if (!result_list) {
        return PyErr_NoMemory();
    }

    return result_list;
}

/**
 * @brief Python wrapper for computing the normalized similarity matrix W.
 *
 * Given two Python lists of lists representing square matrices A (similarity)
 * and D (diagonal degree), this function computes
 *     W = D^{-1/2} * A * D^{-1/2}
 * by delegating to the C routine compute_normalized_similarity_matrix.
 *
 * @param self  Unused.
 * @param args  Tuple containing (A, D) as described above.
 * 
 * @return      New reference to a Python list-of-lists representing W on success;
 *              NULL on failure with a Python exception set.
 */
static PyObject* py_compute_norm(PyObject* self, PyObject* args) {
    PyObject* sim_matrix_list;
    PyObject* ddg_matrix_list;
    PyObject* result_list;
    double** sim_matrix;
    double** degree_matrix;
    double** norm_sim_matrix;
    int n;

    if (!PyArg_ParseTuple(args, "OO", &sim_matrix_list, &ddg_matrix_list)) {
        return NULL;
    }

    sim_matrix = pylist_to_c_matrix(sim_matrix_list, &n, &n);
    degree_matrix = pylist_to_c_matrix(ddg_matrix_list, &n, &n);
    if (!sim_matrix || !degree_matrix) {
        matrix_data_free(sim_matrix, n);
        matrix_data_free(degree_matrix, n);
        return PyErr_NoMemory();
    }
    
    compute_normalized_similarity_matrix(&norm_sim_matrix, sim_matrix, degree_matrix, n);
    matrix_data_free(sim_matrix, n);
    matrix_data_free(degree_matrix, n);
    if (!norm_sim_matrix) {
        return PyErr_NoMemory();
    }

    result_list = c_matrix_to_pylist(norm_sim_matrix, n, n);
    matrix_data_free(norm_sim_matrix, n);
    if (!result_list) {
        return PyErr_NoMemory();
    }

    return result_list;
}

/**
 * @brief Python wrapper for running Symmetric NMF on W with an initial H.
 *
 * Expects four arguments:
 *   1) W   : Python list of lists representing an n * n normalized similarity matrix.
 *   2) H0  : Python list of lists representing an initial factor matrix of shape n * k.
 *   3) k   : int, the target rank (number of columns in H).
 *   4) eps : double, convergence tolerance for the update rule.
 *
 * @param self  Unused.
 * @param args  Tuple containing (W, H0, k, eps) as described above.
 *
 * @return New reference to a Python list of lists (final H) on success; NULL on failure.
 *         On failure, a Python exception is set (e.g., TypeError on bad args,
 *         MemoryError on allocation failure).
 */
static PyObject* py_compute_symnmf(PyObject* self, PyObject* args) {
    PyObject* norm_matrix_list;
    PyObject* H_list;
    PyObject* result_list;
    double** norm_matrix;
    double** H;
    double epsilon;
    int n, k;

    if (!PyArg_ParseTuple(args, "OOid" , &norm_matrix_list, &H_list, &k, &epsilon)) {
        return NULL;
    }

    norm_matrix = pylist_to_c_matrix(norm_matrix_list, &n, &n);
    H = pylist_to_c_matrix(H_list, &n, &k);
    if (!norm_matrix || !H) {
        matrix_data_free(norm_matrix, n);
        matrix_data_free(H, n);
        PyErr_NoMemory();
        return NULL;
    }
    
    symnmf_iterate_H_until_convergence(&H, norm_matrix, n, k, MAX_ITER, epsilon);
    if (!H) {
        matrix_data_free(norm_matrix, n);
        PyErr_NoMemory();
        return NULL;
    }

    result_list = c_matrix_to_pylist(H, n, k);
    matrix_data_free(norm_matrix, n);
    matrix_data_free(H, n);
    if (!result_list) {
        PyErr_NoMemory();
        return NULL;
    }

    return result_list;
}

/* Module method table */
static PyMethodDef SymNMFMethods[] = {
    {"sym", py_compute_sym, METH_VARARGS, "Compute the similarity matrix A from data matrix X."},
    {"ddg", py_compute_ddg, METH_VARARGS, "Compute the diagonal degree matrix D from similarity matrix A."},
    {"norm", py_compute_norm, METH_VARARGS, "Compute the normalized similarity matrix W from A and D."},
    {"symnmf", py_compute_symnmf, METH_VARARGS, "Perform Symmetric NMF on normalized similarity matrix W."},
    {NULL, NULL, 0, NULL}  /* Sentinel */ 
};

/* Module definition */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    "C extension for SymNMF utilities: sym, ddg, norm, and symnmf.",
    -1,
    SymNMFMethods
};

/* Module initialization */
PyMODINIT_FUNC PyInit_symnmf(void) {
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m) {
        return NULL;
    }
    return m;
}
