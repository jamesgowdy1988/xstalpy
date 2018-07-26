// Author: James A Gowdy
// File: common_utils.h
// Date: 2014
// Contact: jamesgowdy@hotmail.co.uk

#ifndef COMMON_UTILS_H // include guard
#define COMMON_UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <sys/time.h>

/*****************************************************************************
 *
 *  Debugging and print macro
 *
 *****************************************************************************/

#ifdef QUIET
#define PRINTF(MSG, ...)
#define LOGINFO(MSG, ...)
#else
#define PRINTF(MSG, ...) printf("> " MSG "\n", ##__VA_ARGS__); fflush(stdout);
#define LOGINFO(MSG, ...) printf("[%s:%d in %s] " MSG "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); fflush(stdout);
#endif

#ifdef VERBOSE
#define DEBUG(MSG, ...) LOGINFO(MSG, ##__VA_ARGS__);
#else
#define DEBUG(MSG, ...)
#endif

/*****************************************************************************
 *
 *  Error checking
 *
 *****************************************************************************/

#define CHECK(A, MSG, ...) if (!(A)) { LOGINFO(MSG, ##__VA_ARGS__); goto error; }

#define CHECK_NOGOTO(A, MSG, ...) if (!(A)) { LOGINFO(MSG, ##__VA_ARGS__); }

/*****************************************************************************
 *
 *  Math macros
 *
 *****************************************************************************/

// maximum value for double precision floating point
#define INFINTE DBL_MAX

// positive modulo (assumes dim is always positive)
#define MOD(x, dim) x<0 ? (x)%dim+dim : (x)%dim

// convert degrees to radians because al, be and ga should be in radians
#define RADIANS(deg) deg * M_PI / 180.0

// convert radians to degrees
#define DEGREES(rad) rad * 180.0 / M_PI

// round to n decimal places - involves expensive power operation
#define ROUND(x, n) round(pow(10, n)*x)/pow(10, n)

// swap values
#define SWAP(x, y, tmp) { tmp = x; x = y; y = tmp; }

// some small value used to account for rounding error
#define TOLERANCE 1e-9

// return true if x is with within +/- 1e-9 of y
#define APPROX(x, y) fabs(x - y) < TOLERANCE

// magnitude of a complex number implemented as an array of length 2
#define MAGNITUDE(cmplx) sqrt(cmplx[0]*cmplx[0] + cmplx[1]*cmplx[1])

/*****************************************************************************
 *
 *  Stats
 *
 *****************************************************************************/

// compute the arithmetic mean
static inline double calc_mean(double n, double x[restrict])
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum/n;
}

// compute the sample variance
static inline double calc_var(double n, double x[restrict])
{
    double sumsq = 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sumsq += x[i]*x[i];
        sum += x[i];
    }
    return (sumsq - sum*sum/n)/(n-1);
}

// compute the sample standard deviation
static inline double calc_sd(double n, double x[restrict])
{
    return sqrt(calc_var(n, x));
}

// compute the standard error of the (sample) mean
static inline double calc_sem(double n, double x[restrict])
{
    return sqrt(calc_var(n, x)/n);
}

/*****************************************************************************
 *
 *  Min and max
 *
 *****************************************************************************/

// return the higher of two values
#define MAX(x, y) x < y ? y : x

// return the lower of two values
#define MIN(x, y) x < y ? x : y

// find the [min, max, absmin and absmax] values and note the index where they occur
static inline void find_minmax(double a[restrict], const int n, double minmax[4], unsigned idx[4])
{
    double min = a[0], max = a[0];
    double val = fabs(a[0]);
    double absmin = val, absmax = val;
    unsigned idxmin = 0, idxmax = 0, idxabsmin = 0, idxabsmax = 0;

    for (unsigned i = 1; i < n; i++) {
        val = a[i];
        if (val < min) {
            min = val;
            idxmin = i;
        }
        if (val > max) {
            max = val;
            idxmax = i;
        }
        val = fabs(a[i]);
        if (val < absmin) {
            absmin = val;
            idxabsmin = i;
        }
        if (val > absmax) {
            absmax = val;
            idxabsmax = i;
        }
    }

    minmax[0] = min; minmax[1] = max; minmax[2] = absmin; minmax[3] = absmax;
    idx[0] = idxmin; idx[1] = idxmax; idx[2] = idxabsmin; idx[3] = idxabsmax;
}

/*****************************************************************************
 *
 *  Vectors algebra
 *
 *****************************************************************************/

// calculate the (3)-row vector euclidean norm
#define NORM(v) sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

// normalize a (3)-row vector
#define UNIT(v, tmp) { tmp = NORM(v); v[0] /= tmp; v[1] /= tmp; v[2] /= tmp; }

// dot product macro for two (3)-row vectors
#define DOT(u, v) (u[0]*v[0] + u[1]*v[1] + u[2]*v[2])

// cross product macro for two (3)-vectors
#define CROSS(u, v, uxv) { \
    uxv[0] = u[1]*v[2] - u[2]*v[1]; \
    uxv[1] = u[2]*v[0] - u[0]*v[2]; \
    uxv[2] = u[0]*v[1] - u[1]*v[0]; }

 /*****************************************************************************
 *
 *  Approximations
 *  http://stackoverflow.com/questions/3380628/fast-arc-cos-algorithm
 *  http://http.developer.nvidia.com/Cg/index_stdlib.html
 *
 *****************************************************************************/

// approximate value of acos, without using native acos
static inline double approx_acos(double x)
{
    double negate, ret;
    negate = (double) (x<0);
    x = fabs(x);
    ret = -0.0187293;
    ret *= x;
    ret += 0.0742610;
    ret *= x;
    ret -= 0.2121144;
    ret *= x;
    ret += 1.5707288;
    ret *= sqrt(1.0-x);
    ret -= 2 * negate * ret;
    return negate * 3.14159265358979 + ret;
}

/*****************************************************************************
 *
 *  Random numbers
 *
 *****************************************************************************/

// seed for random number generation using the microseconds since epoch
static inline int random_seed()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int seed = tv.tv_sec*1000000+tv.tv_usec;
    srand(seed);
    return seed;
}

// get a random floating point number between 0 and 1
static inline double random_double()
{
    return rand()/(double) RAND_MAX;
}

/*****************************************************************************
 *
 *  Strings
 *
 *****************************************************************************/

// fix strncpy by ensure c-string is always NULL-terminated
#define STRNCPY(dest, src, n) { strncpy(dest, src, n); dest[n] = '\0'; }

// get a NULL-terminated sub-string given a source string and destination buffer
static inline void substr(char *arr, char *sub, int start, int end) 
{
    memset(sub, 0, strlen(sub)); // empty sub
    CHECK(end>start, "end < start"); 
    size_t size = end-start;
    memcpy(sub, &arr[start], size); 
    sub[size] = '\0';
    return;
error:
    exit(1);
}

// trim white space from the start and end of a string
static inline int trimstr(char *arr, char *sub) 
{
    memset(sub, 0, strlen(sub)); // empty sub
    char *start = arr;
    char *end = arr + strlen(arr) - 1; // -1 for '\0'

    // trim leading white space
    while (isspace(*start)) {
        start++;
    }

    // if all spaces the end should be null
    if (start == end) {
        *sub = '\0';
        return 0;
    }

    // trim trailing white space 
    while(end > start && isspace(*end)) {
        end--;
    }

    // copy memory in the untrimmed region
    size_t size = end-start+1;
    memcpy(sub, start, size); // +1 because inclusive  

    // make sure NULL terminated
    CHECK(end-start < strlen(arr), "SENTINEL: gone too far?");
    sub[size] = '\0';

    return strlen(sub);
error:
    exit(1);
}

#endif