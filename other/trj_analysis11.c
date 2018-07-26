// Author: James A Gowdy
// File: trj_analysis.c
// Contact: jamesgowdy@hotmail.co.uk
// use P0 everywhere

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "common_utils.h"
#include "trj_analysis.h"
#include "matrix.h"

/*
void rodrigues_matrix(double R[restrict 3][3], double theta, double x, double y, double z)
{
    double norm = sqrt(x*x + y*y + z*z);
    x /= norm; y /= norm; z /= norm;

    // double outer[3][3] = {{x*x, x*y, x*z}, {y*x, y*y, y*z}, {z*x, z*y, z*z}};
    double ident[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double skew[3][3] = {{0, -z, y}, {z, 0, -x}, {-y, x, 0}};
    double skew2[3][3];
    MATRIXMUL(skew, skew, skew2);
    
    theta = RADIANS(theta);
    double c = cos(theta);
    double s = sin(theta);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R[i][j] = ident[i][j] + s*skew[i][j] + (1-c)*skew2[i][j];
            // R[i][j] = c*ident[i][j] + s*skew[i][j] + (1-c)*outer[i][j];
        }
    }

    double det = DET(R);
    CHECK(ROUND(det, 4) == 1, "Non unity determinant, det(R) = %f", det);
    return ;
error:
    exit(1);
}
*/

static inline void cubic_equation(double C[restrict][3], double x[3])
{
    double tmp, b, c, d;

    // characteristic equation for 3x3 matrix
    b = -TRACE(C);
    c = C[0][0]*C[1][1]-C[0][1]*C[1][0] +
        C[0][0]*C[2][2]-C[0][2]*C[2][0] +
        C[1][1]*C[2][2]-C[1][2]*C[2][1];
    d = -DET(C);
    PRINTF("coeffs: %f %f %f", b, c, d);

    /* initial root of the monoic cubic by Newton-Raphson method
    xnew = 10; 
    do {
        xold = xnew;
        xsqu = xold*xold;
        f = xold*xsqu + b*xsqu + c*xold + d;
        fprime = xsqu + 2*b*xold + c;
        xnew = xold - f/fprime;
    } while(fabs(xnew-xold) > 1e-4); 
    x[0] = xnew;
    */
    
    // solution to the monic cubic equation using cardano's method
    tmp = (27*d - 9*b*c + 2*b*b*b)/27;
    x[0] = 2*cbrt(-tmp/2) - b/3;
    PRINTF("initial root: %f", x[0]);

    // carry out synthetic division
    b = b + x[0];
    c = c + b*x[0];
    tmp = d + c*x[0];

    // solve remaining quadratic equation
    tmp = sqrt(b*b - 4*c);
    x[1] = (-b + tmp)/2;
    x[2] = (-b - tmp)/2;

    // reverse bubble sort so x[0] > x[1] > x[2]
    if (x[0] < x[1]) SWAP(x[0], x[1], tmp);
    if (x[1] < x[2]) SWAP(x[1], x[2], tmp);
    if (x[0] < x[1]) SWAP(x[0], x[1], tmp);

    return;
}

static inline void LDUP_decomp(double LDUP[restrict][3], double x[3])
{
    unsigned tmpi, P[3];
    double tmp, e[3];

    // carry out partial pivotting
    P[0] = ARGMAX(LDUP[0]);
    P[1] = 1;
    P[2] = 2;
    P[P[0]] = 0;

    // shuffle remaining columns based on middle row
    if (fabs(LDUP[1][P[2]]) > fabs(LDUP[1][P[1]])) SWAP(P[1], P[2], tmpi);
    // PRINTF("pivots: %d %d %d", P0, P1, P2);

    /* apply partial pivoting by swapping columns, i.e. (C-lam*I)*P
    LDUP[0][0] = LDU[0][P0]; LDUP[0][1] = LDU[0][P1]; LDUP[0][2] = LDU[0][P2];
    LDUP[1][0] = LDU[1][P0]; LDUP[1][1] = LDU[1][P1]; LDUP[1][2] = LDU[1][P2];
    LDUP[2][0] = LDU[2][P0]; LDUP[2][1] = LDU[2][P1]; LDUP[2][2] = LDU[2][P2];

    PRINTF("MP=");
    PRINTMATRIX(LDUP);
    */

    // LU decomposition
    // A00 = U00 // A10 = L10U00       // A20 = L20U00
    // A01 = U01 // A11 = L10U01 + U11 // A21 = L20U01 + L21U11
    // A02 = U02 // A12 = L10U02 + U12 // A22 = L20U02 + L21U12 + U22
    LDUP[1][P[0]] /= LDUP[0][P[0]];
    LDUP[2][P[0]] /= LDUP[0][P[0]];
    LDUP[1][P[1]] -= LDUP[1][P[0]]*LDUP[0][P[1]];
    LDUP[2][P[1]]  = (LDUP[2][P[1]]-LDUP[2][P[0]]*LDUP[0][P[1]])/LDUP[1][P[1]];
    LDUP[1][P[2]] -= LDUP[1][P[0]]*LDUP[0][P[2]];
    LDUP[2][P[2]] -= LDUP[2][P[0]]*LDUP[0][P[2]]+LDUP[2][P[1]]*LDUP[1][P[2]];
    PRINTF("LU=");
    PRINTMATRIX(LDUP);

    // LDU decomposition
    LDUP[0][P[1]] /= LDUP[0][P[0]];
    LDUP[0][P[2]] /= LDUP[0][P[0]];
    LDUP[1][P[2]] /= LDUP[1][P[1]];

    // cartesian basis vector
    e[0] = 0; 
    e[1] = 0; 
    e[2] = 0; 
    e[ARGMIN_DIAG(LDUP)] = 1;

    // back-substitution for LDU: xi = (yi-sum[j>i]Uij*xj)/Uii
    x[P[2]] = e[2];
    x[P[1]] = e[1] - LDUP[1][P[2]]*x[P[2]];
    x[P[0]] = e[0] - LDUP[0][P[1]]*x[P[1]] - LDUP[0][P[2]]*x[P[2]];

    // PRINTF("Px: [%f, %f, %f]", VT[0], VT[1], Px[2]);

    // e = PT*P*x/|x|
    tmp = NORM(x);
    x[0] /= tmp;
    x[1] /= tmp;
    x[2] /= tmp;
}

double align_kabsch(const int n, double A[restrict n][3], double B[restrict n][3])
{
    random_seed();
    A[0][0] = random_double();

    double ATB[3][3], C[3][3], LDU[3][3];
    double LDUP[3][3], VT[3][3], S[3], UT[3][3], R[3][3], e[3];
    double tmp, b, c, d, xold, xsqu, xnew, dx, x[3], theta, f, fprime;
    unsigned i, j, k; // P[3];
    unsigned size = 9*sizeof(double);

    memset(ATB, 0, size);

    /* calculate X.T*Y  make sure compiled with -O3
    for (k = 0; k < n; k++) {
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                ATB[i][j] += A[k][i]*B[k][j];
            }
        }
    } 
    */

    for (k = 0; k < n; k ++) {
        ATB[0][0] += A[k][0]*B[k][0];
        ATB[0][1] += A[k][0]*B[k][1];
        ATB[0][2] += A[k][0]*B[k][2];
        ATB[1][0] += A[k][1]*B[k][0];
        ATB[1][1] += A[k][1]*B[k][1];
        ATB[1][2] += A[k][1]*B[k][2];
        ATB[2][0] += A[k][2]*B[k][0];
        ATB[2][1] += A[k][2]*B[k][1];
        ATB[2][2] += A[k][2]*B[k][2];
    }

    GRAM(ATB, C);
    
    PRINTF("ATB =");
    PRINTMATRIX(ATB);
    PRINTF("C =");
    PRINTMATRIX(C);

    cubic_equation(C, x);
    PRINTF("roots: %f %f %f", x[0], x[1], x[2]);

    // singular values
    // S[0] = sqrt(MAX(x[0], 0));
    // S[1] = sqrt(MAX(x[1], 0));
    // S[2] = sqrt(MAX(x[2], 0));
    // PRINTF("S: [%f, %f, %f]", S[0], S[1], S[2]);

    k = (fabs(x[0]) > 1e-4) + (fabs(x[1]) > 1e-4) + (fabs(x[2]) > 1e-4);
    PRINTF("rank: %d", k);

    // calculate eigenvector for largest and smallest eigenvalues
    for (i = 0; i < 3; i++) {

        if (i != 1) {

        // linear equations for eigenvectors: ATA - lam*I = 0
        memcpy(LDUP, C, size); // LDU
        LDUP[0][0] -= x[i];
        LDUP[1][1] -= x[i];
        LDUP[2][2] -= x[i];

        PRINTF("M=");
        PRINTMATRIX(LDUP);

        LDUP_decomp(LDUP, VT[i]);

        PRINTF("LDUP=")
        PRINTMATRIX(LDUP);

        // PRINTF("x: [%f, %f, %f]", VT[i][0], VT[i][1], VT[i][2]);
        }
    }

    // enforce orthogonality by taking cross vector
    CROSS(VT[2], VT[0], VT[1]);

    switch (k) {
        // AT*B = U*S*VT -> U = AT*B*V*SI -> U = AT*B*V -> UT = VT*BTA
        // UT[i][j] = sum_k VT[i][k]*BTA[k][j]
        // UT[i][j] = sum_k VT[i][k]*ATB[j][k]
        case 3:
            MATRIXMUL_RIGHTT(VT, ATB, UT);
            //tmp = sqrt(UT[0][0]*UT[0][0]+UT[0][1]*UT[0][1]+UT[0][2]*UT[0][2]);
            //UT[0][0] /= tmp;
            //UT[0][1] /= tmp;
            //UT[0][2] /= tmp;
            //tmp = sqrt(UT[1][0]*UT[1][0]+UT[1][1]*UT[1][1]+UT[1][2]*UT[1][2]);
            //UT[1][0] /= tmp;
            //UT[1][1] /= tmp;
            //UT[1][2] /= tmp;
            //tmp = sqrt(UT[2][0]*UT[2][0]+UT[2][1]*UT[2][1]+UT[2][2]*UT[2][2]);
            //UT[2][0] /= tmp;
            //UT[2][1] /= tmp;
            //UT[2][2] /= tmp;
            UNIT(UT[0], tmp);
            UNIT(UT[1], tmp);
            UNIT(UT[2], tmp);
            break;
    /*
        case 2:
            UT[0][0] = DOT(VT[0], ATB[0]);
            UT[0][1] = DOT(VT[0], ATB[1]);
            UT[0][2] = DOT(VT[0], ATB[2]);
            UT[1][0] = DOT(VT[1], ATB[0]);
            UT[1][1] = DOT(VT[1], ATB[1]);
            UT[1][2] = DOT(VT[1], ATB[2]);
            UNIT(UT[0], tmp);
            UNIT(UT[1], tmp);
            CROSS(UT[0], UT[1], UT[2]);
            break;
        case 1:
            UT[0][0] = DOT(VT[0], ATB[0]);
            UT[0][1] = DOT(VT[0], ATB[1]);
            UT[0][2] = DOT(VT[0], ATB[2]);
            UNIT(UT[0], tmp);
            e[0] = 0; e[1] = 0; e[2] = 0; e[ARGMIN(UT[0])] = 1;
            CROSS(UT[0], e, UT[1]);
            CROSS(UT[0], UT[1], UT[2]);
            break; */
        // case 0:
            // CHECK(0, "Zero rank matrix?");
    }

    MATRIXMUL_LEFTT(UT, VT, R);

    PRINTF("UT=");
    PRINTMATRIX(UT);
    PRINTF("VT=");
    PRINTMATRIX(VT);
    PRINTF("R=");
    PRINTMATRIX(R);

    tmp = TRACE(R);
    theta = approx_acos((tmp-1)/2);
    tmp = 2*sin(theta);
    x[0] = (R[2][1]-R[1][2])/tmp;
    x[1] = (R[0][2]-R[2][0])/tmp;
    x[2] = (R[1][0]-R[0][1])/tmp;
    
    PRINTF("theta = %f", DEGREES(theta));
    return DEGREES(theta);
//error:
//    exit(1);
}

int main() {
    /*
    double X[30][3], Y[30][3];

    // 30 random coordinates
    for (int i = 0; i < 30; i++) {
        for (int j = 0; j < 3; j++) {
            X[i][j] = random_double();
        }
    }

    // define a 20 degree rotation around x+y
    double R[3][3];
    rodrigues_matrix(R, 20.0, 1.0, 1.0, 0.0);
    PRINTMATRIX(R);

    // 30 rotated coordinates
    for (int i = 0; i < 30; i++) {
        for (int j = 0; j < 3; j++) {
            Y[i][j] = R[j][0]*X[i][0] + R[j][1]*X[i][1] + R[j][2]*X[i][2];
        }
    } 
    */

    double A[30][3] = 
        {{    0.225,    0.268,    0.243},
         {    0.432,    0.393,    0.430},
         {    0.313,    0.536,    0.454},
         {    0.359,    0.670,    0.815},
         {    0.682,    0.239,    0.621},
         {    0.818,    0.892,    0.977},
         {    0.727,    0.553,    0.501},
         {    0.525,    0.927,    0.806},
         {    0.460,    0.744,    0.383},
         {    0.445,    0.314,    0.931},
         {    0.307,    0.884,    0.967},
         {    0.482,    0.507,    0.558},
         {    0.719,    0.275,    0.885},
         {    0.135,    0.226,    0.279},
         {    0.756,    0.068,    0.342},
         {    0.799,    0.644,    0.844},
         {    0.392,    0.590,    0.133},
         {    0.537,    0.424,    0.826},
         {    0.452,    0.766,    0.921},
         {    0.338,    0.035,    0.423},
         {    0.711,    0.572,    0.202},
         {    0.509,    0.468,    0.481},
         {    0.631,    0.722,    0.681},
         {    0.758,    0.410,    0.711},
         {    0.132,    0.299,    0.479},
         {    0.241,    0.611,    0.875},
         {    0.576,    0.543,    0.075},
         {    0.931,    0.122,    0.633},
         {    0.631,    0.830,    0.245},
         {    0.537,    0.839,    0.338}};

    double B[30][3] = 
        {{    0.285,    0.207,    0.239},
         {    0.535,    0.290,    0.395},
         {    0.430,    0.419,    0.480},
         {    0.566,    0.464,    0.841},
         {    0.819,    0.102,    0.477},
         {    1.056,    0.654,    0.936},
         {    0.843,    0.437,    0.428},
         {    0.732,    0.720,    0.855},
         {    0.561,    0.643,    0.428},
         {    0.666,    0.093,    0.844},
         {    0.558,    0.633,    1.048},
         {    0.617,    0.372,    0.530},
         {    0.920,    0.074,    0.724},
         {    0.205,    0.155,    0.284},
         {    0.818,    0.006,    0.155},
         {    0.998,    0.445,    0.756},
         {    0.430,    0.552,    0.173},
         {    0.733,    0.228,    0.749},
         {    0.684,    0.534,    0.941},
         {    0.432,   -0.058,    0.324},
         {    0.755,    0.528,    0.156},
         {    0.624,    0.353,    0.442},
         {    0.798,    0.554,    0.662},
         {    0.919,    0.249,    0.585},
         {    0.253,    0.178,    0.491},
         {    0.464,    0.388,    0.912},
         {    0.593,    0.526,    0.062},
         {    1.060,   -0.006,    0.400},
         {    0.696,    0.765,    0.279},
         {    0.628,    0.748,    0.391}};

    // array_printer(30, 3, X);
    // array_printer(30, 3, Y);

    //SILENCE = 0;

    long ii;
    double c = 0.;
    for (ii = 0; ii < 1000*1000*50; ii++) {
        c += align_kabsch(30, A, B);
        exit(1);
    }
    // printf("%ld and %f\n", ii, c);

    return 0;
}




/*
int read_dcdfile_example(char input_path[])
{
    DcdHeader head = parse_dcdheader(input_path);
    FILE *fp = fopen(input_path, "r");
    fseek(fp, head.start_address, SEEK_SET);

    size_t size_int = sizeof(int);
    size_t size_float = sizeof(float);
    size_t offset = sizeof(int) + head.xstal_head + sizeof(int);

    int tmp[2];
    int i = 0, rc = 0;
    float x[head.natoms];
    float y[head.natoms];
    float z[head.natoms];

    while (1) {
        if (head.xstal_head) {
            fseek(fp, offset, SEEK_CUR);
        }           
        rc = fread(tmp, size_int, 1, fp);
        CHECK(rc == 1, "End of file after %d frames", i);
        CHECK(tmp[0] == head.coord_head, "Unexpected coord size");
        fread(x, size_float, head.natoms, fp);
        fread(tmp, size_int, 2, fp);
        fread(y, size_float, head.natoms, fp);
        fread(tmp, size_int, 2, fp);
        fread(z, size_float, head.natoms, fp);
        rc = fread(tmp, size_int, 1, fp);
        CHECK(rc == 1, "Unexpected end of file after %d frames", i);
        i++;
    }

error: // fall-through
    fclose(fp);
    return 0;
} */
