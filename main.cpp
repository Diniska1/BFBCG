#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <chrono>

// compile + run:
// g++ main.cpp -lopenblas -llapack && ./a.out

extern "C" { // Multiply matrixes
    extern int dnrm2_(unsigned int*, double *, unsigned int *);
}

extern "C" { // Multiply matrixes
    extern int dgemm_(char *, char *, unsigned int *, unsigned int *, unsigned int *, double *, 
                    double *, unsigned int *, double *, unsigned int *, double *, double *, unsigned int *);
}

extern "C" { // QR DECOMPOSITION
    extern int dgeqrf_(unsigned int *, unsigned int *, double *, unsigned int *, double *, double *, int *, int *);
}

extern "C" { // COPY VECTOR
    extern int dcopy_(unsigned int *, double *, int *, double *, int *);
}

extern "C" { // Compute matrix Q in QR
    extern int dorgqr_(unsigned int *, unsigned int *, unsigned int *, double *, unsigned int *, double *, double *, int *, int *);
}

extern "C" { // PIVOTING QR DECOMPOSITION
    extern int dgeqp3_(unsigned int *, unsigned int *, double *, unsigned int *, int *, double *, double *, int *, int *);
}

extern "C" { // LU FACTORIZATION
    extern int dgetrf_(unsigned int *, unsigned int *, double *, unsigned int *, int *, int *);
}

extern "C" { // SOLVING LINEAR SYSTEM
    extern int dgetrs_(char *, unsigned int *, unsigned int *, double *, unsigned int *, int *, double *, unsigned int *, int *);
}

extern "C" { // SWAP COLUMNS
    extern int dswap_(unsigned int *, double *, int *, double *, int *);
}


double * fill_in_x(unsigned int n, unsigned int s)
{
    double * x = new double[n * s];
    for(int i = 0; i < n * s; i++) x[i] = 0;

    return x;
}

double * fill_in_b(unsigned int n, unsigned int s)
{
    double * b = new double [n * s];
    for (int i = 0; i < n; ++i){
        for(int j = 0; j < s; ++j) {
            b[j * n + i] = (j + i) % n + 1;
        }
    }
    return b;
}

double * fill_in_A(unsigned int n)
{
    double * A = new double [n * n];
    for (int i = 0; i < n; ++i) {
        A[i * n + i] = 4;
        if (i + 1 < n) {
            A[i * n + i + 1] = -1;
            A[(i + 1) * n + i] = -1;
        }
    }
    return A;
}


void print_matrix(double * ary, unsigned int n, unsigned int s, int dist = 15) 
{
    std::cout << std::endl << std::setw(dist);
    for(int i = 0; i < n; i++) {
        
        for(int j = 0; j < s; j++) {
            std::cout  << ary[j * n + i] << std::setw(dist);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


unsigned int orth(double * p, double * q, double * r, unsigned int n, unsigned int s, double eps)
{
/*
HOW FUNCTION WORKS:
1) P = QR (QR decomposition for matrix P)
2) R = Q1 * R1 * P1 (QR decomposition with pivoting for matrix R, P1 is the permutation matrix)
3) Searching index of element in main diagonal of matrix R1, which less than eps. Let that index is equal cur_rank
4) Result matrix is equal to product of matrixes: P = Q * Q2, where matrix Q2 contains first cur_rank columns of matrix Q1
*/

    for(int i = 0; i < s * s; i++) {
        r[i] = 0;
    }

    int inc = 1;
    unsigned int size = n * s;
    dcopy_(&size, p, &inc, q, &inc);


    int lwork = 3 * n + 1;
    double * work = new double[3 * n + 1];
    double * tau = new double [n];
    int info = 1;

    dgeqrf_(&n, &s, q, &n, tau, work, &lwork, &info);   // P = QR

    // filling R
    for(int i = 0; i < s; i++) {                         
        for(int j = i; j < s; j++) {
            r[j * s + i] = q[j * n + i];
        }
    }
    
    dorgqr_(&n, &s, &s, q, &n, tau, work, &lwork, &info); // compute Q for P = QR

    //std::cout << "\n=========== R in P = QR ===========\n"; print_matrix(r, s, s);
    int * jpvt = new int[s];
    dgeqp3_(&s, &s, r, &s, jpvt, tau, work, &lwork, &info); // pivoting QR for R
    //std::cout << "\n=========== pivoting for R ===========\n"; print_matrix(r, s, s);

    unsigned int cur_rank = s;
    for(int i = 0 ; i < s; ++i) {
        if (std::abs(r[i * s + i]) < eps) {
            cur_rank = i;
            break;
        }
    }
    if (cur_rank == 0) {
        return 0;
    }

    dorgqr_(&s, &s, &s, r, &s, tau, work, &lwork, &info); // compute Q in R = QR
    //std::cout << "\n=========== Q in pivoting for R ===========\n"; print_matrix(r, s, s);


    char trans = 'N'; double alph = 1.0; double bet = 0.0;
    dgemm_(&trans, &trans, &n, &cur_rank, &s, &alph, q, &n, r, &s, &bet, p, &n);
    return cur_rank;
    
}


double cheb_norm(double * r, int col, unsigned int n)
{
    double max = 0.0;
    for(int i = 0; i < n; ++i) {
        if(std::abs(r[col * n + i]) > max) max = std::abs(r[col * n + i]);
    }
    return max;
}

void swap_cols(double * x, double * r, int i, unsigned int j, int * swaps, unsigned int n, unsigned int s)
{
// Swapping i-th and j-th columns of matrixes X and R
    int inc = 1;
    dswap_(&n, &r[i * n], &inc, &r[j * n], &inc);
    dswap_(&n, &x[i * n], &inc, &x[j * n], &inc);
    if(j != 0) {
        unsigned int t = swaps[i];
        swaps[i] = swaps[j];
        swaps[j] = t;
    }

}

void swap_back(double * x, int * swaps, unsigned int n, unsigned int s)
{
// swaps: (example)
// [3 1 4 2] -> [4 1 3 2] -> [2 1 3 4] -> [1 2 3 4]
    int inc = 1;
    for(unsigned int i = 1; i <= s; ++i) {
        if(swaps[i - 1] != i - 1) {
            dswap_(&n, &x[(i - 1) * n], &inc, &x[swaps[i - 1] * n], &inc);

            unsigned int t = swaps[i - 1];
            swaps[i - 1] = swaps[swaps[i - 1]];
            swaps[t] = t;
            i -= 1;
        }
        
        
    }
}

void check_allocate(double * r, double * alpha, double * beta, double * p, 
                    double * workQ, double * q, double * PQ, double * workR, int * ipiv)
{
    if(!r) std::cout << "R bad allocate\n";
    if(!alpha) std::cout << "alpha bad allocate\n";
    if(!beta) std::cout << "beta bad allocate\n";
    if(!p) std::cout << "P bad allocate\n";
    if(!workQ) std::cout << "workQ bad allocate\n";
    if(!workR) std::cout << "workR bad allocate\n";
    if(!q) std::cout << "Q bad allocate\n";
    if(!PQ) std::cout << "PQ bad allocate\n";
    if(!ipiv) std::cout << "ipiv bad allocate\n";
}

double max_elem(unsigned int n, unsigned int s, double * workQ)
{
    double max = 0.0;
    for(unsigned int i = 0; i < n * s; ++i) {
        if(std::abs(workQ[i]) > max) {
            max = std::abs(workQ[i]);
        }
    }
    return max;
}

unsigned int max(unsigned int a, unsigned int b) {
    if (a > b) return a;
    return b;
}















int main() {
    unsigned int n, s;
    std::cout << "Size of matrix: ";
    std::cin >> n;
    std::cout << "Amount right sides: ";
    std::cin >> s;
    unsigned int cur_s = s;

    if(n < s) {
        std::cout << "WARNING: n should be more or equal than s\n";
        return 1;
    }



    double * a = fill_in_A(n);
    double * b = fill_in_b(n, s);          // RIGHT SIDE size n x s
    double * x = fill_in_x(n, s);          // SOLUTION size n x s
    double * r = new double[n * s];        // RESUDIAL size n x s
    double * alpha = new double[s * s];    // matrix alpha size cur_rank x s
    double * beta = new double [s * s];    // matrix beta size cur_rank x s
    double * p = new double[n * s];        // MATRIX P size n x cur_rank
    double * workQ = new double[n * s];    // work memory
    double * q = new double[n * s];        // MATRIX Q = A * P, size n x cur_rank
    double * PQ = new double [s * s];      // MATRIX PQ = P^T * Q, size cur_rank x s
    double * workR = new double[s * s];    // work memory
    int *ipiv = new int[s];                // for dgetrf_ and dgetrs_
    int *swaps = new int[s];               // For store swaps


    unsigned int size = n * s; 
    double alp = -1.0, bet = 1.0;           // coefficients for dgemm_ and dgetrs_
    char trans = 'N', trans2 = 'T';         // chars for dgemm_ and dgetrs_
    int inc = 1, info;                      // for BLAS, LAPACK functions
    double eps = 1e-5, eps_orth = 1e-12;     // Precision
    unsigned int converged_for = 0;

    check_allocate(r,alpha,beta,p,workQ,q,PQ,workR,ipiv);

    dcopy_(&size, b, &inc, r, &inc);
    
    dgemm_(&trans, &trans, &n, &s, &n, &alp, a, &n, x, &n, &bet, r, &n); // R = B - A * X_0

    dcopy_(&size, r, &inc, p, &inc); // preconditioner M is identical matrix

    unsigned int cur_rank = 
    orth(p, workQ, workR, n, s, eps_orth);


    for(int i = 0 ; i < s; ++i) {
        ipiv[i] = i + 1;
        swaps[i] = i;
    }
    
    

// ===================================================
// ================== MAIN CYCLE =====================
// ===================================================

    auto start_time = std::chrono::steady_clock::now();
    unsigned int max_iterations = n;
    for(unsigned int i = 0; i < max_iterations; ++i) {
        //start_cycle:
        unsigned int max_s_rank = max(cur_s, cur_rank);
        //std::cout << "========================= i = " << i << " ========================\n";

        // Q_i = A * P_i
        bet = 0.0; alp = 1.0;
        dgemm_(&trans, &trans, &n, &cur_rank, &n, &alp, a, &n, p, &n, &bet, q, &n); 
        //std::cout << "-------- Q ---------"; print_matrix(q, n, cur_rank);


        //calculate alpha_i
        alp = 1.0; bet = 0.0;
        dgemm_(&trans2, &trans, &cur_rank, &cur_rank, &n, &alp, p, &n, q, &n, &bet, PQ, &cur_rank);
        dgemm_(&trans2, &trans, &cur_rank, &cur_s, &n, &alp, p, &n, r, &n, &bet, alpha, &max_s_rank);
        dgetrf_(&cur_rank, &cur_rank, PQ, &cur_rank, ipiv, &info);
        dgetrs_(&trans, &cur_rank, &cur_s, PQ, &cur_rank, ipiv, alpha, &max_s_rank, &info);
        //std::cout << "-------- alpha ---------"; print_matrix(alpha, cur_rank, cur_rank);


        // X_i+1 = X_i + P_i * alpha_i
        alp = 1.0; bet = 1.0;
        dgemm_(&trans, &trans, &n, &cur_s, &cur_rank, &alp, p, &n, alpha, &max_s_rank, &bet, x, &n); 
        // std::cout << "-------- X ---------"; print_matrix(x, n, cur_rank);


        // R_i+1 = R_i - Q_i * alpha_i
        alp = -1.0; bet = 1.0;
        dgemm_(&trans, &trans, &n, &cur_s, &cur_rank, &alp, q, &n, alpha, &max_s_rank, &bet, r, &n); 
        //std::cout << "-------- R ---------"; print_matrix(r, n, cur_s);
        
        // check columns in R
        start_cycle:
        for(int j = 0; j < cur_s; ++j) { 
            if(cheb_norm(r, j, n) < eps) {
                swap_cols(x, r, j, cur_s - 1, swaps, n, s);
                cur_s--;
                
                if(!cur_s) { // CONVERGED!
                    converged_for = i + 1;
                    goto after_cycle;
                    break;
                }
                goto start_cycle;
            }
        }
        

        //calculate beta_i
        alp = 1.0; bet = 0.0;
        dgemm_(&trans2, &trans, &cur_rank, &cur_s, &n, &alp, q, &n, r, &n, &bet, beta, &max_s_rank);
        dgetrs_(&trans, &cur_rank, &cur_s, PQ, &cur_rank, ipiv, beta, &max_s_rank, &info);
        //std::cout << "-------- beta ---------"; print_matrix(beta, cur_rank, cur_rank);


        // P_i+1 = R_i+1 + P_i * beta_i
        dcopy_(&size, r, &inc, workQ, &inc);
        alp = -1.0; bet = 1.0;
        dgemm_(&trans, &trans, &n, &cur_rank, &cur_rank, &alp, p, &n, beta, &cur_rank, &bet, workQ, &n); 
        dcopy_(&size, workQ, &inc, p, &inc);
        // std::cout << "-------- P_i+1 ---------"; print_matrix(p, n, cur_rank);

        // orth(P_i+1);
        cur_rank = orth(p, workQ, workR, n, cur_rank, eps_orth);
        if (cur_rank == 0) goto after_cycle;
        //std::cout << "-------- orth(P_i+1) ---------"; print_matrix(p, n, cur_rank);

    }
    after_cycle:
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "TIME: " <<  double(elapsed_ns.count()) / 1000.0 << " s\n";

    

    swap_back(x,swaps,n,s);

    int check_for_correct = 0;
    if(check_for_correct) {
        alp = 1.0; bet = 0.0;
        dgemm_(&trans, &trans, &n, &s, &n, &alp, a, &n, x, &n, &bet, workQ, &n);
        std::cout << "================= A * X ==================\n";
        print_matrix(workQ, n, s);
        std::cout << "================= X ==================\n";
        print_matrix(x, n, s);
        
    }

    dcopy_(&size, b, &inc, workQ, &inc);
    alp = -1.0; bet = 1.0;
    dgemm_(&trans, &trans, &n, &s, &n, &alp, a, &n, x, &n, &bet, workQ, &n);
    double res = max_elem(n, s, workQ);
    std::cout << "Max element of B - AX: \t\t\t" << res << std::endl;
    std::cout << "Algorithm converged for iterations: \t" << converged_for << std::endl; 

    return 0;
}


// compile + run:
// g++ NAMEFILE.cpp -lopenblas -llapack && ./a.out