#ifndef _TRJ_ANALYSIS_H
#define _TRJ_ANALYSIS_H

#define ARGMAX(a) ((fabs(a[0]) > fabs(a[1])) && (fabs(a[0]) > fabs(a[2]))) ? 0 : (fabs(a[1]) > fabs(a[2]) ? 1 : 2)

#define ARGMIN(a) ((fabs(a[0]) < fabs(a[1])) && (fabs(a[0]) < fabs(a[2]))) ? 0 : (fabs(a[1]) < fabs(a[2]) ? 1 : 2)

#define ARGMIN_DIAG(a) ((fabs(a[0][0]) < fabs(a[1][1])) && (fabs(a[0][0]) < fabs(a[2][2]))) ? 0 : (fabs(a[1][1]) < fabs(a[2][2]) ? 1 : 2)

#define DETSYM(A) \
     (A[0][0]*A[1][1]*A[2][2] \
    + 2*A[0][1]*A[1][2]*A[0][2] \
    - A[0][2]*A[1][1]*A[0][2] \
    - A[0][1]*A[0][1]*A[2][2] \
    - A[0][0]*A[1][2]*A[1][2])

const double SIGN[4][4] = {{ 1,-1, 1, -1}, {-1, 1, -1, 1}, { 1,-1, 1, -1}, {-1, 1, -1, 1}};

#define COFACTOR_3x3(A, i1, i2, j1, j2) \
     (A[i1][j1]*A[i2][j2] - A[i1][j2]*A[i2][j1])

#define COFACTOR_4x4(A, i0, i1, i2, j0, j1, j2) \
     (A[i0][j0]*A[i1][j1]*A[i2][j2] \
    + A[i0][j1]*A[i1][j2]*A[i2][j0] \
    + A[i0][j2]*A[i1][j0]*A[i2][j1] \
    - A[i0][j2]*A[i1][j1]*A[i2][j0] \
    - A[i0][j1]*A[i1][j0]*A[i2][j2] \
    - A[i0][j0]*A[i1][j2]*A[i2][j1])

#define ADJCOL(j, j0, j1, j2) \
    x[0] = SIGN[0][j]*COFACTOR_4x4(B, 1, 2, 3, j0, j1, j2); \
    x[1] = SIGN[1][j]*COFACTOR_4x4(B, 0, 2, 3, j0, j1, j2); \
    x[2] = SIGN[2][j]*COFACTOR_4x4(B, 0, 1, 3, j0, j1, j2); \
    x[3] = SIGN[3][j]*COFACTOR_4x4(B, 0, 1, 2, j0, j1, j2); \
    norm2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];


/*
static inline double min3(double a[restrict]) 
{
    double a0 = fabs(a[0]), a1 = fabs(a[1]), a2 = fabs(a[2]);
    double tmp = MIN(a0, a1);
    return MIN(tmp, a2);
}

static inline double max3(double a[restrict]) 
{
    double a0 = fabs(a[0]), a1 = fabs(a[1]), a2 = fabs(a[2]);
    double tmp = MAX(a0, a1);
    return MAX(tmp, a2);
}

static inline double min4(double a[restrict]) 
{
    double a0 = fabs(a[0]), a1 = fabs(a[1])), a2 = fabs(a[2]), a3 = fabs(a[3]);
    double tmp = MIN(a0, a1);
    tmp = MIN(tmp, a2);
    return MIN(tmp, a3);
}

static inline double max4(double a[restrict]) 
{
    double a0 = fabs(a[0]), a1 = fabs(a[1])), a2 = fabs(a[2]), a3 = fabs(a[3]);
    double tmp = MAX(a0, a1);
    tmp = MAX(tmp, a2);
    return MAX(tmp, a3);
}

#define VARMAT_DECLARE(A) \
    double A##_##00, A##_## 01, A##_##02; \
    double A##_##10, A##_## 11, A##_##12; \
    double A##_##20, A##_## 21, A##_##22;

#define VARMAT_INIT(A) \
    double A##_##00 = 0, A##_##01 = 0, A##_##02 = 0; \
    double A##_##10 = 0, A##_##11 = 0, A##_##12 = 0; \
    double A##_##20 = 0, A##_##21 = 0, A##_##22 = 0;

// trace of a (3,3) arrays
#define VARMAT_TRACE(A) (A##_##00 + A##_##11 + A##_##22)

// determinant of a (3,3) arrays
#define VARMAT_DET(A) \
     (A##_##00 * A##_##11 * A##_##22 \
    + A##_##01 * A##_##12 * A##_##20 \
    + A##_##02 * A##_##10 * A##_##21 \
    - A##_##02 * A##_##11 * A##_##20 \
    - A##_##01 * A##_##10 * A##_##22 \
    - A##_##00 * A##_##12 * A##_##21)

#define VARMAT_TO_ARRAY(A, B) \
    B[0][0] = A##_##00; B[0][1] = A##_##01; B[0][2] = A##_##02; \
    B[1][0] = A##_##10; B[1][1] = A##_##11; B[1][2] = A##_##12; \
    B[2][0] = A##_##20; B[2][1] = A##_##21; B[2][2] = A##_##22;

#define VARMAT_ARGMIN_DIAG(A) fabs(A##_##00) < fabs(A##_##11) ? \
    (fabs(A##_##00) < fabs(A##_##22) ? 0 : 2) : \
    (fabs(A##_##11) < fabs(A##_##22) ? 1 : 2)

#define VARMAT_COVAR(A, ATA) \
    ATA##_##00 = A##_##00*A##_##00 + A##_##10*A##_##10 + A##_##20*A##_##20; \
    ATA##_##01 = A##_##00*A##_##01 + A##_##10*A##_##11 + A##_##20*A##_##21; \
    ATA##_##02 = A##_##00*A##_##02 + A##_##10*A##_##12 + A##_##20*A##_##22; \
    ATA##_##10 = ATA##_##01; \
    ATA##_##11 = A##_##01*A##_##01 + A##_##11*A##_##11 + A##_##21*A##_##21; \
    ATA##_##12 = A##_##01*A##_##02 + A##_##11*A##_##12 + A##_##21*A##_##22; \
    ATA##_##20 = ATA##_##02; \
    ATA##_##21 = ATA##_##12; \
    ATA##_##22 = A##_##02*A##_##02 + A##_##12*A##_##12 + A##_##22*A##_##22;

#define VARMATMUL_RIGHTT(A, BT, AB) { \
    AB[0][0] = A[0][0]*BT##_##00 + A[0][1]*BT##_##01 + A[0][2]*BT##_##02; \
    AB[0][1] = A[0][0]*BT##_##10 + A[0][1]*BT##_##11 + A[0][2]*BT##_##12; \
    AB[0][2] = A[0][0]*BT##_##20 + A[0][1]*BT##_##21 + A[0][2]*BT##_##22; \
    AB[1][0] = A[1][0]*BT##_##00 + A[1][1]*BT##_##01 + A[1][2]*BT##_##02; \
    AB[1][1] = A[1][0]*BT##_##10 + A[1][1]*BT##_##11 + A[1][2]*BT##_##12; \
    AB[1][2] = A[1][0]*BT##_##20 + A[1][1]*BT##_##21 + A[1][2]*BT##_##22; \
    AB[2][0] = A[2][0]*BT##_##00 + A[2][1]*BT##_##01 + A[2][2]*BT##_##02; \
    AB[2][1] = A[2][0]*BT##_##10 + A[2][1]*BT##_##11 + A[2][2]*BT##_##12; \
    AB[2][2] = A[2][0]*BT##_##20 + A[2][1]*BT##_##21 + A[2][2]*BT##_##22; }
*/
#endif
