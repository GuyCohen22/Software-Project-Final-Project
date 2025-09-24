#ifndef SYMNMF_H
#define SYMNMF_H

/* Free a double** with 'rows' rows (safe on NULL). */
void matrix_data_free(double **data, int rows);

/* Build A (n * n) from data (n * dim). On success sets *sim_matrix; caller frees with matrix_data_free(*sim_matrix, n). */
void compute_similarity_matrix(double ***sim_matrix, double **data, int n, int dim);

/* Build diagonal degree matrix D (n * n) from A (n * n). On success sets *degree_matrix; caller frees with matrix_data_free(*degree_matrix, n). */
void compute_diagonal_degree_matrix(double ***degree_matrix, double **sim_matrix, int n);

/* Compute W = D^{-1/2} * A * D^{-1/2} (all n * n). On success sets *out_norm_sim_matrix; caller frees with matrix_data_free(*out_norm_sim_matrix, n). */
void compute_normalized_similarity_matrix(double ***out_norm_sim_matrix, double **sim_matrix, double **degree_matrix, int n);

/* Run SymNMF on H (rows * k) with W (rows * rows) until convergence or MAX_ITER. On success *H points to the final matrix. */
void symnmf_iterate_H_until_convergence(double ***H, double **W, int rows, int k, const int MAX_ITER, const double EPSILON);

#endif
