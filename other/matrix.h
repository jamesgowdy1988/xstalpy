// Author: James A Gowdy
// File: common_utils.h
// Date: 2014
// Contact: jamesgowdy@hotmail.co.uk

#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdio.h>

/*****************************************************************************
 *
 *  Pretty printers
 *
 *****************************************************************************/

// print a (M,N) array for python
static inline void array_printer(size_t mrows, size_t ncols, double A[mrows][ncols])
{
    printf("[");
    for (int i = 0; i < mrows; i++) {
        if (i == 0) printf("[");
        else printf(" [");
        for (int j = 0; j < ncols; j++) {
            if (j != ncols-1) printf("%9.3f,", A[i][j]);
            else printf("%9.3f", A[i][j]);
        }
        if (i != mrows-1) printf("],\n");
        else printf("]]\n");
    }
}

// print a (M,N) array for c
static inline void carray_printer(size_t mrows, size_t ncols, double A[mrows][ncols])
{
    printf("{");
    for (int i = 0; i < mrows; i++) {
        if (i == 0) printf("{");
        else printf(" {");
        for (int j = 0; j < ncols; j++) {
            if (j != ncols-1) printf("%9.3f,", A[i][j]);
            else printf("%9.3f", A[i][j]);
        }
        if (i != mrows-1) printf("},\n");
        else printf("}}\n");
    }
}

// print a (3,3) array
#ifdef QUIET
#define PRINTMATRIX(A)
#else
#define PRINTMATRIX(A) array_printer(3, 3, A)
#endif

/*****************************************************************************
 *
 *  Misc matrix operations
 *
 *****************************************************************************/

// trace of a (3,3) matrix array
#define TRACE(A) (A[0][0]+A[1][1]+A[2][2])

// determinant of a (3,3) matrix arrays
#define DET(A) \
     (A[0][0]*A[1][1]*A[2][2] \
    + A[0][1]*A[1][2]*A[2][0] \
    + A[0][2]*A[1][0]*A[2][1] \
    - A[0][2]*A[1][1]*A[2][0] \
    - A[0][1]*A[1][0]*A[2][2] \
    - A[0][0]*A[1][2]*A[2][1])

/*****************************************************************************
 *
 *  Matrix multiplication
 *
 *****************************************************************************/

// multiply two (3,3) matrix arrays
#define MATRIXMUL(A, B, AB) { \
    AB[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]; \
    AB[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1]; \
    AB[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2]; \
    AB[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0]; \
    AB[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1]; \
    AB[1][2] = A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2]; \
    AB[2][0] = A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0]; \
    AB[2][1] = A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1]; \
    AB[2][2] = A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2]; }

// multiply two (3,3) matrix arrays, equiv to TRANSPOSE(AT, A); MATRIXMUL(A, B, AB)
#define MATRIXMUL_LEFTT(AT, B, AB) { \
    AB[0][0] = AT[0][0]*B[0][0] + AT[1][0]*B[1][0] + AT[2][0]*B[2][0]; \
    AB[0][1] = AT[0][0]*B[0][1] + AT[1][0]*B[1][1] + AT[2][0]*B[2][1]; \
    AB[0][2] = AT[0][0]*B[0][2] + AT[1][0]*B[1][2] + AT[2][0]*B[2][2]; \
    AB[1][0] = AT[0][1]*B[0][0] + AT[1][1]*B[1][0] + AT[2][1]*B[2][0]; \
    AB[1][1] = AT[0][1]*B[0][1] + AT[1][1]*B[1][1] + AT[2][1]*B[2][1]; \
    AB[1][2] = AT[0][1]*B[0][2] + AT[1][1]*B[1][2] + AT[2][1]*B[2][2]; \
    AB[2][0] = AT[0][2]*B[0][0] + AT[1][2]*B[1][0] + AT[2][2]*B[2][0]; \
    AB[2][1] = AT[0][2]*B[0][1] + AT[1][2]*B[1][1] + AT[2][2]*B[2][1]; \
    AB[2][2] = AT[0][2]*B[0][2] + AT[1][2]*B[1][2] + AT[2][2]*B[2][2]; }

// multiply two (3,3) matrix arrays, equiv to TRANSPOSE(BT, B); MATRIXMUL(A, B, AB)
#define MATRIXMUL_RIGHTT(A, BT, AB) { \
    AB[0][0] = A[0][0]*BT[0][0] + A[0][1]*BT[0][1] + A[0][2]*BT[0][2]; \
    AB[0][1] = A[0][0]*BT[1][0] + A[0][1]*BT[1][1] + A[0][2]*BT[1][2]; \
    AB[0][2] = A[0][0]*BT[2][0] + A[0][1]*BT[2][1] + A[0][2]*BT[2][2]; \
    AB[1][0] = A[1][0]*BT[0][0] + A[1][1]*BT[0][1] + A[1][2]*BT[0][2]; \
    AB[1][1] = A[1][0]*BT[1][0] + A[1][1]*BT[1][1] + A[1][2]*BT[1][2]; \
    AB[1][2] = A[1][0]*BT[2][0] + A[1][1]*BT[2][1] + A[1][2]*BT[2][2]; \
    AB[2][0] = A[2][0]*BT[0][0] + A[2][1]*BT[0][1] + A[2][2]*BT[0][2]; \
    AB[2][1] = A[2][0]*BT[1][0] + A[2][1]*BT[1][1] + A[2][2]*BT[1][2]; \
    AB[2][2] = A[2][0]*BT[2][0] + A[2][1]*BT[2][1] + A[2][2]*BT[2][2]; }

// multiply a (3,3) matrix by its transpose, AT*A
#define GRAM(A, ATA) \
    ATA[0][0] = A[0][0]*A[0][0] + A[1][0]*A[1][0] + A[2][0]*A[2][0]; \
    ATA[0][1] = A[0][0]*A[0][1] + A[1][0]*A[1][1] + A[2][0]*A[2][1]; \
    ATA[0][2] = A[0][0]*A[0][2] + A[1][0]*A[1][2] + A[2][0]*A[2][2]; \
    ATA[1][0] = ATA[0][1]; \
    ATA[1][1] = A[0][1]*A[0][1] + A[1][1]*A[1][1] + A[2][1]*A[2][1]; \
    ATA[1][2] = A[0][1]*A[0][2] + A[1][1]*A[1][2] + A[2][1]*A[2][2]; \
    ATA[2][0] = ATA[0][2]; \
    ATA[2][1] = ATA[1][2]; \
    ATA[2][2] = A[0][2]*A[0][2] + A[1][2]*A[1][2] + A[2][2]*A[2][2];

/*  
 *  covraince matrix is can either be a feature covariance matrix (M, M)
 *  or a data covraince matrix (N, N) depending on whether we compute
 *  (X-E[X])(Y-E[Y])^T or (X-E[X])^T(Y-E[Y])
 */

#endif
