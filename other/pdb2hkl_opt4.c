/*******************************************************************************

pdb2hkl - short program for Fhkl calculation with minimal dependecies

# install FFTW
sudo port install fftw # MacOS
sudo apt-get install fftw3 # Ubuntu

# compilation
gcc pdb2hkl_opt.c -std=c99 -Ofast -lfftw3 -L /opt/local/lib -I /opt/local/include -o pdb2hkl

# compilation options
-DTO_FILE # writes h, k, l, A, phi to out.txt
-DQUIET # turns off logging info and error messages

# example run parameters
./pdb2hkl # gives call 
./pdb2hkl 1AO6.pdb 1.0 0.001 17.0 # -> 0.2ms
./pdb2hkl 1AO6.pdb 3.0 0.01 17.0 # -> +1.2s

# trouble-shooting
ulimit -S -s unlimited

*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h> // isspace
#include <complex.h>
#include <fftw3.h> // must install FFTW

/*******************************************************************************
*
* Math and error macros plus pretty print formatting
*
*******************************************************************************/

#define OUTPUT out.hkl

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define PI 3.14159265358979323846

#ifndef QUIET
#define LOGINFO(MSG, ...) \
    printf("[%s:%d in %s] " MSG "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); fflush(stdout);
#else
#define LOGINFO(MSG, ...)
#endif

#define ASSERT_WARN(A, MSG, ...) \
    if (!(A)) { LOGINFO(MSG, ##__VA_ARGS__); }
#define ASSERT_ERROR(A, MSG, ...) \
    if (!(A)) { LOGINFO(MSG, ##__VA_ARGS__); goto error; }
#define ASSERT_EXIT(A, MSG, ...) \
    if (!(A)) { LOGINFO(MSG, ##__VA_ARGS__); exit(1); }

void print_matrix(double A[3][3]) {
    printf("[[%8.5f, %8.5f, %8.5f],\n", A[0][0], A[0][1], A[0][2]);
    printf(" [%8.5f, %8.5f, %8.5f],\n", A[1][0], A[1][1], A[1][2]);
    printf(" [%8.5f, %8.5f, %8.5f]]\n", A[2][0], A[2][1], A[2][2]);
}

/*******************************************************************************
*
* Data: common element Cromer-Mann data, chiral spacegroups and laue group info
*
*******************************************************************************/

#define NELEMS 16
#define NSPG 65

// Cromer-Mann A coefficents
double ACM[][4] = {
    {17.1789, 5.2358, 5.6377, 3.9851}, // BR
    {2.31, 1.02, 1.5886, 0.865}, // C
    {8.6266, 7.3873, 1.5899, 1.0211}, // CA
    {11.4604, 7.1964, 6.2556, 1.6455}, // CL
    {12.2841, 7.3409, 4.0034, 2.3488}, // CO
    {13.338, 7.1676, 5.6158, 1.6735}, // CU
    {11.7695, 7.3573, 3.5222, 2.3045}, // FE
    {0.489918, 0.262003, 0.196767, 0.049879}, // H
    {26.9049, 17.294, 14.5583, 3.63837}, // HO
    {20.1472, 18.9949, 7.5138, 2.2735}, // I
    {8.2186, 7.4398, 1.0519, 0.8659}, // K
    {12.2126, 3.1322, 2.0125, 1.1663}, // N
    {4.7626, 3.1736, 1.2674, 1.1128}, // NA
    {3.0485, 2.2868, 1.5463, 0.867}, // O
    {6.4345, 4.1791, 1.78, 1.4908}, // P
    {6.9053, 5.2034, 1.4379, 1.5863}, // S
};

// Cromer-Mann B coefficents
double BCM[][4] = {
    {2.1723, 16.5796, 0.2609, 41.4328}, // BR
    {20.8439, 10.2075, 0.5687, 51.6512}, // C
    {10.4421, 0.6599, 85.7484, 178.437}, // CA
    {0.0104, 1.1662, 18.5194, 47.7784}, // CL
    {4.2791, 0.2784, 13.5359, 71.1692}, // CO
    {3.5828, 0.247, 11.3966, 64.8126}, // CU
    {4.7611, 0.3072, 15.3535, 76.8805}, // FE
    {20.6593, 7.74039, 49.5519, 2.20159}, // H
    {2.07051, 0.19794, 11.4407, 92.6566}, // HO
    {4.347, 0.3814, 27.766, 66.8776}, // I
    {12.7949, 0.7748, 213.187, 41.6841}, // K
    {0.0057, 9.8933, 28.9975, 0.5826}, // N
    {3.285, 8.8422, 0.3136, 129.424}, // NA
    {13.2771, 5.7011, 0.3239, 32.9089}, // O
    {1.9067, 27.157, 0.526, 68.1645}, // P
    {1.4679, 22.2151, 0.2536, 56.172}, // S
};

// Cromer-Mann C coefficents
double CCM[] = {
    2.9557, // BR
    0.2156, // C
    1.3751, // CA
    -9.5574, // CL
    1.0118, // CO
    1.191, // CU
    1.0369, // FE
    0.001305, // H
    4.56796, // HO
    4.0712, // I
    1.4228, // K
    -11.529, // N
    0.676, // NA
    0.2508, // O
    1.1149, // P
    0.8669, // S
};

// order elements in the Cromer-Mann tables
char *elements[] = {"BR", "C", "CA", "CL", "CO", "CU", "FE", "H", "HO", "I", "K",
                    "N", "NA", "O", "P", "S"};

// chiral space group ITC codes
int spg_itcchiral[] = {1, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22, 23, 24, 75, 76,
                    77, 78, 79, 80, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 143,
                    144, 145, 146, 149, 150, 151, 152, 153, 154, 155, 168, 169,
                    170, 171, 172, 173, 177, 178, 179, 180, 181, 182, 195, 196,
                    197, 198, 199, 207, 208, 209, 210, 211, 212, 213, 214};

// chiral space group HM symbols
char *spg_hmchiral[] = {"P 1", "P 1 2 1", "P 1 21 1", "C 1 2 1", "P 2 2 2",
                    "P 2 2 21", "P 21 21 2", "P 21 21 21", "C 2 2 21", "C 2 2 2",
                    "F 2 2 2", "I 2 2 2", "I 21 21 21", "P 4", "P 41", "P 42",
                    "P 43", "I 4", "I 41", "P 4 2 2", "P 4 21 2", "P 41 2 2",
                    "P 41 21 2", "P 42 2 2", "P 42 21 2", "P 43 2 2",
                    "P 43 21 2","I 4 2 2", "I 41 2 2", "P 3", "P 31", "P 32",
                    "R 3", "P 3 1 2", "P 3 2 1", "P 31 1 2", "P 31 2 1",
                    "P 32 1 2", "P 32 2 1", "R 3 2", "P 6", "P 61", "P 65",
                    "P 62", "P 64", "P 63", "P 6 2 2", "P 61 2 2", "P 65 2 2",
                    "P 62 2 2", "P 64 2 2", "P 63 2 2", "P 2 3", "F 2 3",
                    "I 2 3", "P 21 3", "I 21 3", "P 4 3 2", "P 42 3 2",
                    "F 4 3 2", "F 41 3 2", "I 4 3 2", "P 43 3 2", "P 41 3 2",
                    "I 41 3 2"};

// chiral space group laue group
// code = (0 = -1, 1 = 2/m, 2 = m m m, 3 = 4/m, 4 = 4/m m m, 5 = -3, 6 = -3 1 m,
//  7 = -3 m 1, 8 = 6/m, 9 = 6/m m m, 10 = m -3, 11 = m -3 m)
int spg_laue[] = {0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4,
                  4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 7, 6, 7, 6, 7, 7, 8, 8,
                  8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11,
                  11, 11, 11, 11, 11};

// chiral space group unique wedge hkl limit
// code = (0 = -limit-limit, 1 = 0-limit, 2 = h-limit, 3 = k-limit)
int laue_limits[12][3] = {{0, 0, 1}, // -1 (coverage = 1/2)
                          {0, 1, 1}, // 2/m (coverage = 1/4)
                          {1, 1, 1}, // m m m (or 2/m m??) (coverage = 1/8)
                          {1, 1, 1}, // 4/m (coverage = 1/8)
                          {1, 2, 1}, // 4/m m m (coverage = 1/16)
                          {1, 1, 0}, // -3 (coverage = 1/6)
                          {1, 2, 0}, // -3 1 m (coverage = 1/12)
                          {1, 1, 1}, // -3 m 1 (coverage = 1/12)
                          {1, 1, 1}, // 6/m (coverage = 1/12)
                          {1, 2, 1}, // 6/m m m (coverage = 1/24)
                          {1, 2, 2}, // m -3 (coverage = 1/24)
                          {1, 2, 3}}; // m -3 m (coverage = 1/48)

/*******************************************************************************
*
* PDB typedefs and memory allocation
*
*******************************************************************************/

typedef struct Symop {
    double R[3][3];
    double t[3];
} Symop;

typedef struct Atoms {
    double a, b, c, al, be, ga;
    int natoms;
    int nelems;
    int nsym;
    int spg;
    int laue;
    double Bmin;
    double *restrict x;
    double *restrict y;
    double *restrict z;
    double *restrict B;
    double *restrict n;
    int *restrict e;
    int *restrict newe;
    int *restrict new2old;
    int *restrict old2new;
    Symop *sym; // note sym->R is equiv to sym[0].R
} Atoms;

Atoms *create_Atoms(int natoms, int nsym) {
    // allocate memory for Atoms struct
    Atoms *atoms = malloc(sizeof(Atoms));
    atoms->natoms = natoms;
    atoms->nelems = 0;
    atoms->nsym = nsym;
    atoms->spg = -1;

    atoms->x = malloc(natoms*sizeof(double));
    atoms->y = malloc(natoms*sizeof(double));
    atoms->z = malloc(natoms*sizeof(double));
    atoms->B = malloc(natoms*sizeof(double));
    atoms->n = malloc(natoms*sizeof(double));
    atoms->e = malloc(natoms*sizeof(int));
    atoms->newe = malloc(natoms*sizeof(int));

    atoms->new2old = malloc(NELEMS*sizeof(int));
    atoms->old2new = malloc(NELEMS*sizeof(int));
    atoms->sym = malloc(nsym*sizeof(Symop));
    atoms->Bmin = 1e10;

    for (int i = 0; i < NELEMS; i++) {
        atoms->new2old[i] = -1;
        atoms->old2new[i] = -1;
    }
    return atoms;
}

void free_Atoms(Atoms *atoms) {
    // free memory for Atoms struct
    free(atoms->x);
    free(atoms->y);
    free(atoms->z);
    free(atoms->B);
    free(atoms->n);
    free(atoms->e);
    free(atoms->newe);
    free(atoms->new2old);
    free(atoms->old2new);
    free(atoms->sym);
}

/*******************************************************************************
*
* PDB parsing and helper functions
*
*******************************************************************************/

void substr(const char *str, char *sub, int start, int end) {
    // get a substring sub = str[start:end]
    size_t size = end-start;
    memcpy(sub, &str[start], size);
    sub[size] = '\0';
}

char *trimstr(char *str) {
    // trim leading and trailing whitespace from a cstring
    while(isspace(*str)) str++;
    if (*str == 0) return str;
    char *end = str + strlen(str) - 1;
    while(end > str && isspace(*end)) end--;
    *(end+1) = 0;
    return str;
}

Atoms *parse_pdb(char *filename) {

    // open existing file for reading
    FILE *file = fopen(filename, "r");
    ASSERT_EXIT(file != NULL, "Error: No file %s", filename);

    // will loop over fixed-length lines of 80 chars (82 because "%80s\n\0")
    int maxlen = 82;

    // declare variables
    double Ri0, Ri1, Ri2, t;
    char line[maxlen], buffer[maxlen], ebuffer[3], spgbuffer[12];
    char *spg;
    int natoms = 0, nsym = 0, isym;

    // do preliminary loop to get size of memory required
    while (fgets(line, maxlen, file) != NULL) {

        // check line length is 80
        ASSERT_EXIT(line[80] == '\n', "Error: check line[%d]: %s", natoms, line);

        // get first 6 characters
        substr(line, buffer, 0, 6);

        // get total number of ATOM and HETATM entries
        if (strcmp(buffer, "ATOM  ") == 0 || strcmp(buffer, "HETATM") == 0) {
            natoms++;

        // read symmetry operators
        } else if (strcmp(buffer, "REMARK") == 0) {
            substr(line, buffer, 0, 18);
            if (strcmp(buffer, "REMARK 290   SMTRY") == 0) {
                sscanf(&line[21], "%d", &isym);
                nsym = MAX(isym, nsym);
            }
        }
    }

    ASSERT_EXIT(natoms > 0, "Error: No ATOM or HETATM records? (natoms = %d)", natoms);
    ASSERT_EXIT(nsym > 0, "Error: No REAMRK 290 record? (nsym = %d)", nsym);

    // reset to beginning of file
    fseek(file, 0, SEEK_SET);

    // allocate memory of struct to be returned
    Atoms *p = create_Atoms(natoms, nsym);

    // read again and loop over fixed-length lines of 80 chars
    int i = 0, j, e;
    while (fgets(line, maxlen, file) != NULL) {

        substr(line, buffer, 0, 6);

        if (strcmp(buffer, "ATOM  ") == 0 || strcmp(buffer, "HETATM") == 0) {

            // read coordinate data (alternative use scanf)
            substr(line, buffer, 30, 38);
            p->x[i] = atof(buffer);
            substr(line, buffer, 38, 46);
            p->y[i] = atof(buffer);
            substr(line, buffer, 47, 54);
            p->z[i] = atof(buffer);
            substr(line, buffer, 55, 60);
            p->n[i] = atof(buffer);
            substr(line, buffer, 60, 66);
            p->B[i] = atof(buffer);
            substr(line, buffer, 76, 78);
            p->Bmin = MIN(p->B[i], p->Bmin);

            // get custom element ID for CM arrays
            sscanf(buffer, "%s", ebuffer);
            p->e[i] = -1;
            for (e = 0; e < NELEMS; e++) {
                if (strcmp(ebuffer, elements[e]) == 0) {
                    p->e[i] = e;
                    break;
                }
            }

            ASSERT_ERROR(p->e[i] != -1, "require '%s' coeffs:\n%s", ebuffer, line);
            ASSERT_ERROR(i < natoms, "i = %d < %d = natoms (eh?)", i, natoms);

            if (p->old2new[e] == -1) {
                p->old2new[e] = p->nelems;
                p->new2old[p->nelems] = e;
                p->nelems++;
            }
            p->newe[i] = p->old2new[e];
            i++;

        } else if (strcmp(buffer, "CRYST1") == 0) {

            // read unit cell parameters
            sscanf(&line[6], "%lf %lf %lf %lf %lf %lf", &p->a, &p->b, &p->c,
                   &p->al, &p->be, &p->ga);

            // convert to radians
            p->al *= PI/180.0;
            p->be *= PI/180.0;
            p->ga *= PI/180.0;

            ASSERT_ERROR(p->a != 0.0, "Error in CRYST1?: %s", line);

            // read/trim HM spacegroup symbol & convert to ITC spacegoup number
            substr(line, spgbuffer, 55, 66);
            spg = trimstr(spgbuffer);

            for (j = 0; j < NSPG; j++) {
                if (strcmp(spg, spg_hmchiral[j]) == 0) {
                    p->spg = spg_itcchiral[j];
                    p->laue = spg_laue[j];
                }
            }

            ASSERT_ERROR(p->spg != -1, "Error: Spacegroup not chiral? (%s)", spg);

        // read symmetry operators
        } else if (strcmp(buffer, "REMARK") == 0) {

            substr(line, buffer, 0, 18);
            if (strcmp(buffer, "REMARK 290   SMTRY") == 0) {

                // read symmetry operators
                sscanf(&line[18], "%d %d %lf %lf %lf %lf", &j, &isym, &Ri0, &Ri1,
                       &Ri2, &t);

                p->sym[isym-1].R[j-1][0] = Ri0;
                p->sym[isym-1].R[j-1][1] = Ri1;
                p->sym[isym-1].R[j-1][2] = Ri2;
                p->sym[isym-1].t[j-1] = t;
            }
        }
    }

    ASSERT_ERROR(p->spg != -1, "Error: No CRYST1 record?");
    LOGINFO("natoms = %d", natoms);
    LOGINFO("nsym = %d", nsym);
    LOGINFO("spg_hm = '%s'", spg);
    LOGINFO("spg_itc = '%d'", p->spg);

    fclose(file);
    return p;

error:
    free_Atoms(p);
    fclose(file);
    exit(1);
}

/*******************************************************************************
*
* Basis transformation
*
*******************************************************************************/

void calc_scale(double a, double b, double c, double al, double be, double ga,
                double S[][3]) {

    // al be ga must be in radians
    double tmp = cos(al)*cos(al) + cos(be)*cos(be) + cos(ga)*cos(ga);
    double V = a*b*c * sqrt(1 - tmp + 2*cos(al)*cos(be)*cos(ga));

    S[0][0] = 1.0/a;
    S[0][1] = -cos(ga)/(a*sin(ga));
    S[0][2] = (b*cos(ga)*c*(cos(al)-cos(be)*cos(ga))/sin(ga)-b*c*cos(be)*sin(ga))/V;

    S[1][0] = 0.0;
    S[1][1] = 1.0/(b*sin(ga));
    S[1][2] = -a*c*(cos(al)-cos(be)*cos(ga))/(V*sin(ga));

    S[2][0] = 0.0;
    S[2][1] = 0.0;
    S[2][2] = a*b*(sin(ga)/V);
}

void calc_invscale(double a, double b, double c, double al, double be, double ga,
                   double SI[][3]) {

    // al be ga must be in radians
    double tmp = cos(al)*cos(al) + cos(be)*cos(be) + cos(ga)*cos(ga);
    double V = a*b*c * sqrt(1 - tmp + 2*cos(al)*cos(be)*cos(ga));

    SI[0][0] = a;
    SI[1][0] = 0.0;
    SI[2][0] = 0.0;

    SI[0][1] = b*cos(ga);
    SI[1][1] = b*sin(ga);
    SI[2][1] = 0.0;

    SI[0][2] = c*cos(be);
    SI[1][2] = c*(cos(al) - cos(be)*cos(ga))/sin(ga);
    SI[2][2] = V/(a*b*sin(ga));
}

/*******************************************************************************
*
* Denisty typedef and memory allocation
*
*******************************************************************************/

typedef struct Grid {
    double ***data; // could restrict
    double *raw;
    int n1;
    int n2;
    int n3;
    double Bextra;
} Grid;

Grid *create_Grid(int n1, int n2, int n3) {

    Grid *p = malloc(sizeof(Grid));
    p->n1 = n1;
    p->n2 = n2;
    p->n3 = n3;

    // raw contiguous memory block
    // double *d = fftw_malloc(n1*n2*n3*sizeof(double)); //see _allocate_rawdata()
    double *d = malloc(n1*n2*n3*sizeof(double));
    //memset(d, 0, n1*n2*n3*sizeof(double));
    ASSERT_ERROR(d != NULL, "memory error");

    memset(d, 0, n1*n2*n3*sizeof(double));
    p->raw = d;

    // alternative 3D data access
    // could restrict in scope (see Performance_Tuning_with_the_RESTRICT_Keyword 4.5)
    double **dd = malloc(n1*n2*sizeof(double*));
    double *** ddd = malloc(n1*sizeof(double**));
    ASSERT_ERROR(dd != NULL, "memory error");
    ASSERT_ERROR(ddd != NULL, "memory error");

    for (int i = 0; i < n1; i++) {
        ddd[i] = dd + n2*i;
    }
    for (int i = 0; i < n1*n2; i++) {
        dd[i] = d + n3*i;
    }
    p->data = ddd;
    return p;

error:
    // if (d) fftw_free(d);
    if (d) free((void*) d);
    if (dd) free((void*) dd);
    if (ddd) free((void*) ddd);
    exit(1);
}

void free_Grid(Grid *p) {
    // fftw_free(p->data[0][0]);
    free((void*) p->data[0][0]); // free rawdata (cast for restrict)
    free((void*) p->data[0]); // free section
    free((void*) p->data);
    free(p);
}

/*******************************************************************************
*
* Density sampling
*
*******************************************************************************/

int calc_griddim(double ucdim, double nearest, double dmin, double invg) {

    // dmin                                    (Fhkl resolution desired)
    // shannon = dmin/2                        (shannon sampling interval)
    // sigma = invg/2                          (oversampling rate)
    // delta = dmin/(2*sigma) = dmin/invg      (map sampling interval [ang/vox])
    // rate = na/a = invg/dmin = 1/(dmin*g)    (map sampling rate or multiplier)
    // na = a*rate                             (grid dimensions)
    //    = a*invg/dmin 
    //    = a*2*sigma/dmin 
    //    = a*sigma/shannon

    double rate = invg/dmin; // [voxs/ang]
    double vox = rate * ucdim; // [voxs/ang*ang = voxs]

    return round(vox/nearest)*nearest; // [voxels]
}

double calc_Bextra(double Q, double dmin, double invg, double Bmin) {

    // example values: Q = 100, dmin = 1.0, invg = 3.0, pdb->Bmin = 0.5
    // dmin                                    (Fhkl resolution desired)
    // shannon = dmin/2                        (shannon sampling interval)
    // sigma = invg/2                          (oversampling rate)
    // delta = dmin/(2*sigma) = dmin/invg      (map sampling interval [ang/vox])
    // dstaralias = 1/delta                    (aliasing distance in recip R3)
    //            = invg/dmin
    //            = 2*sigma/dmin
    //            = 2*sigma*dstarmax

    // derivation of signal to noise (1st alias interference) ratio, Q
    // Fprimary = exp(-B*dstar2/4) * n*exp(i*alpha)
    // Falias   = exp(-B*(1*dstaralias-dstar)^2/4) * n*exp(i*alpha)
    // max(Falias) ~ exp(-B*min(dstaralias-dstar)^2/4)
    //             ~ exp(-B*(dstaralias-dstarmax)^2/4)
    //             ~ exp(-B*(2*sigma - 1)^2*dstarmax2/4)
    //             ~ exp(-B*(4*sigma^2 - 4*sigma + 1)*dstarmax2/4)
    //             ~ exp(-B*sigma*(sigma - 1)*dstarmax2 - B*dstarmax2/4)
    // Q = Fprimary/Falias
    //   = exp(B*sigma*(sigma-1)*dstarmax2)
    // B = ln(Q)/(sigma*(sigma-1)*dstarmax2)
    // B = Beq + Bextra
    // max(Bextra) = ln(Q)/(sigma*(sigma-1)*dstarmax2) - min(Beq)

    // Fraw = exp(-Bextra*dstar2/4) * sum(n*exp(-Beq*dstar2/4)*exp(i*alpha))
    // F = exp(Bextra*dstar2/4) * Fraw

    // ref: ITC B. section 1.3

    ASSERT_EXIT(invg != 2, "Error: invalid Bbase when invg/2 == 1", invg);
    double sigma = 0.5*invg;
    double Bbase = log(Q)*dmin*dmin/(sigma*(sigma-1));
    double Bextra = Bbase - Bmin;

    LOGINFO("Bmin = %f", Bmin);
    LOGINFO("Bbase = %f", Bbase);
    LOGINFO("Bextra = %f", Bextra);

    return Bextra;
}

Grid *pdb2map(double dmin, double invg, double rhocut, double Bextra, 
              Atoms *atoms) {

    // exptable lookup size and error margin (incase d2 > dcut2 slightly)
    const int ntable = 4000;
    const double dcut2_error = 1.2;

    // calculate grid dimensions to nearest 10
    const int na = calc_griddim(atoms->a, 8.0, dmin, invg); 
    const int nb = calc_griddim(atoms->b, 8.0, dmin, invg);
    const int nc = calc_griddim(atoms->c, 8.0, dmin, invg);
    LOGINFO("na, nb, nc = %d %d %d", na, nb, nc);

    // declare/initalize grid (stack/heap speeds are similar)
    Grid *rho = create_Grid(nc, nb, na);

    // world/crystal coordinate basis transformation matrix
    double S[3][3], SI[3][3];
    calc_scale(atoms->a, atoms->b, atoms->c, atoms->al, atoms->be, atoms->ga, S);
    calc_invscale(atoms->a, atoms->b, atoms->c, atoms->al, atoms->be, atoms->ga, SI);

    // world/grid coordinate basis transformation
    double G[3][3], GI[3][3];
    for (int i = 0; i < 3; i++) {
        G[0][i] = S[0][i]*na;
        G[1][i] = S[1][i]*nb;
        G[2][i] = S[2][i]*nc;
        GI[i][0] = SI[i][0]/ (double) na;
        GI[i][1] = SI[i][1]/ (double) nb;
        GI[i][2] = SI[i][2]/ (double) nc;
    }

    // for atom parameters/coordinates
    double n, B, xw, yw, zw, xc, yc, zc;
    int e, xg, yg, zg, xg_corr, yg_corr, zg_corr;

    // for voxel coordinates
    int dxg, dyg, dzg, rxg, ryg, rzg;
    double dxw, dyw, dzw, dzw2, dyw2_dzw2, d2;

    // for sampling box limits
    int dxg_max, dyg_max, dzg_max;
    double dxw_max, dyw_max, dzw_max;

    // for density cut-off radius
    double dcuts[atoms->natoms], dcut, dcut2, step;

    // for density calculation/lookup
    double prho[5], erho[5], rhod, rho0, valmax = -1e10, val, fdx;
    int idx;

    // loop over atoms
    for (int i = 0; i < atoms->natoms; i++) {

        e = atoms->e[i];
        B = atoms->B[i] + Bextra;
        n = atoms->n[i];

        // if (B == 0) B = 15.0;
        ASSERT_EXIT(B > 0, "Error: B[%d] must be positive", i);
        ASSERT_EXIT(n > 0, "Error: n[%d] must be positive", i);

        // pre-calculate model electron density as inverse scattering factor
        prho[4] = n * CCM[e] * pow(4.0*PI/B, 1.5);
        erho[4] = 4.0*PI*PI/B; // B can not be 0
        rho0 = prho[4];
        for (int j = 0; j < 4; j++) {
            prho[j] = n * ACM[e][j] * pow(4*PI/(BCM[e][j]+B), 1.5);
            erho[j] = 4.0*PI*PI/(BCM[e][j]+B);
            rho0 += prho[j];
        }

        // get cutoff radius, for rho(dcut) = rhocut
        step = 1;
        rhod = rho0;
        while (rhod/rho0 > rhocut) {
            dcut = step*dmin/invg;
            dcut2 = dcut*dcut;
            rhod = 0.0;
            for (int j = 0; j < 5; j++) {
                rhod += prho[j] * exp(-erho[j]*dcut2);
            }
            step += 1;
            ASSERT_EXIT(step < 100, "step iteration > 100");
        }
        dcuts[i] = dcut;
        dcut2 *= dcut2_error;
        // LOGINFO("dcut2 = %f", dcut2);

        // get maximum limit needed for the exptable value 
        MAX(valmax, erho[4]*dcut2);
        for (int j = 0; j < 4; j++) {
            valmax = MAX(valmax, erho[j]*dcut2);
        }
    }

    // val/valmax = idx/idxmax -> val*idxmax/valmax = idx -> k = idxmax/valmax
    const double k = (ntable-1.0)/valmax;
    LOGINFO("valmax = %f", valmax)

    // pre-calculated negative exponential table
    double exptable[ntable];
    for (int i = 0; i < ntable; i++) {
        exptable[i] = exp(-i/k);
    }

    // loop over all atom coordinates for density sampling
    for (int i = 0; i < atoms->natoms; i++) {

        // model parameters
        e = atoms->e[i];
        n = atoms->n[i];
        B = atoms->B[i] + Bextra;

        // atom center's world coordinates
        xw = atoms->x[i];
        yw = atoms->y[i];
        zw = atoms->z[i];

        // atom center's crystallographic coordinates
        xc = S[0][0]*xw + S[0][1]*yw + S[0][2]*zw;
        yc = S[1][1]*yw + S[1][2]*zw;
        zc = S[2][2]*zw;

        // atom center's grid coordinates
        xg = (int) round(xc*na);
        yg = (int) round(yc*nb);
        zg = (int) round(zc*nc);

        // with correction for positive modulo when added to sampling box coord
        zg_corr = zg + nc + nc;
        yg_corr = yg + nb + nb;
        xg_corr = xg + na + na;

        // recalculate model electron density as inverse scattering factor
        prho[4] = n * CCM[e] * pow(4.0*PI/B, 1.5);
        erho[4] = k*4.0*PI*PI/B; // * k for table lookup
        rho0 = prho[4];
        for (int j = 0; j < 4; j++) {
            prho[j] = n * ACM[e][j] * pow(4*PI/(BCM[e][j]+B), 1.5);
            erho[j] = k*4.0*PI*PI/(BCM[e][j]+B); // * k for table lookup
            rho0 += prho[j];
        }
        
        // calculated cutoff radius
        dcut = dcuts[i];
        dcut2 = dcut*dcut;

        // calculate sampling box limits (for cubic only need single octant)
        dzg_max = round(G[2][2]*dcut);

        // loop over z-axis (c) of crystallographic grid
        for (dzg = -dzg_max; dzg < dzg_max+1; dzg++) {

            // voxel z grid coordinate
            rzg = (zg_corr+dzg)%nc;

            // world coordinates 
            dzw = GI[2][2]*dzg;
            dzw2 = dzw*dzw;

            // calculate sampling box limits
            // if (dcut2 < dzw2) continue;
            dyw_max = sqrt(dcut2 - dzw2);
            // dyg_max = round(G[1][1]*dyw_max + G[1][2]*dzw);
            // dyg_max *= (dcut2 >= dzw2);
            dyg_max = G[1][1]*dyw_max + G[1][2]*dzw;
            dyg_max = (dcut2 < dzw2) ? 0 : dyg_max;

            // loop over y-axis (b) of crystallographic grid
            for (dyg = -dyg_max; dyg < dyg_max+1; dyg++) {

                // voxel y grid coordinate
                ryg = (yg_corr + dyg) % nb;

                // world coordinates
                dyw = GI[1][1]*dyg + GI[1][2]*dzg;
                dyw2_dzw2 = dyw*dyw + dzw2;

                // calculate sampling box limits
                // if (dcut2 < dyw2_dzw2) continue;
                dxw_max = sqrt(dcut2 - dyw2_dzw2);
                // dxg_max = round(G[0][0]*dxw_max + G[0][1]*dyw + G[0][2]*dzw);
                // dxg_max *= (dcut2 >= dyw2_dzw2);
                dxg_max = G[0][0]*dxw_max + G[0][1]*dyw + G[0][2]*dzw;
                dxg_max = (dcut2 < dyw2_dzw2) ? 0 : dxg_max;

                // loop over x-axis (a) of crystallographic grid
                for (dxg = -dxg_max; dxg < dxg_max+1; dxg++) {

                    // calculate voxel to atom center distance squared
                    dxw = GI[0][0]*dxg + GI[0][1]*dyg + GI[0][2]*dzg;
                    d2 = dxw*dxw + dyw2_dzw2;
                    if (d2 > dcut2) continue;

                    // density at d^2 using exptable lookup (no bound checks)
                    rhod = 0.0;
                    for (int j = 0; j < 5; j++) {
                        idx = (int) (erho[j]*d2);
                        rhod += prho[j] * exptable[idx]; // exp(-erho[j]*d2); 
                    }

                    // voxel x grid coordinate
                    rxg = (xg_corr + dxg) % na;

                    // atom's density contribution to this voxel
                    rho->data[rzg][ryg][rxg] += rhod; // * (d2 < dcut2);
                }
            }
        }
    }

    rho->Bextra = Bextra;
    return rho;
}

/*******************************************************************************
*
* Refln typedef
*
*******************************************************************************/

typedef struct {
    int h, k, l;
    double A, phi, dstar;
} Refln;

typedef struct {
    int nrefln;
    Refln *F;
    int hmax;
    int hmin;
    int *kmax;
    int *kmin;
    int **lmax;
    int **lmin;
} Reflns;

void transpose(double A[3][3], double AT[3][3]) {
    AT[0][0] = A[0][0];
    AT[0][1] = A[1][0];
    AT[0][2] = A[2][0];

    AT[1][0] = A[0][1];
    AT[1][1] = A[1][1];
    AT[1][2] = A[2][1];

    AT[2][0] = A[0][2];
    AT[2][1] = A[1][2];
    AT[2][2] = A[2][2];
}

Reflns *create_Reflns(double dmin, Atoms *pdb) {

    double dstarmax2 = 1.0/(dmin*dmin);
    Reflns *p = malloc(sizeof(Reflns));

    double a = pdb->a;
    double b = pdb->b;
    double c = pdb->c;
    double al = pdb->al;
    double be = pdb->be;
    double ga = pdb->ga;

    double S[3][3], SI[3][3], SIT[3][3];
    calc_invscale(a, b, c, al, be, ga, S);
    calc_invscale(a, b, c, al, be, ga, SI);
    // transpose(SI, SIT);

    int nrefln = 0;
    int h, k, kmin, kmax, lmin, lmax, hi, ki;
    double hw, kw, hw2, kw2, kwmax, lwmax;

    int kmins[3];
    int lmins[4];
    kmins[1] = 0;
    lmins[1] = 0;

    int limit_h = laue_limits[pdb->laue][0];
    int limit_k = laue_limits[pdb->laue][1];
    int limit_l = laue_limits[pdb->laue][2];

    // h = SIT[0][0]*hw + SIT[0][1]*kw + SIT[0][2]*lw;
    // k = SIT[1][0]*hw + SIT[1][1]*kw + SIT[1][2]*lw;
    // l = SIT[2][0]*hw + SIT[2][1]*kw + SIT[2][2]*lw;

    int hmax = (int) (SI[0][0]/dmin + 1);
    int hmin = (limit_h == 0) ? -hmax : 0;

    p->hmax = hmax;
    p->hmin = hmin;
    p->kmin = malloc((hmax-hmin)*sizeof(int));
    p->kmax = malloc((hmax-hmin)*sizeof(int));
    p->lmin = malloc((hmax-hmin)*sizeof(int*));
    p->lmax = malloc((hmax-hmin)*sizeof(int*));

    // count reflectsions and save limits
    for (h = hmin; h <= hmax; h++) {
        hw = S[0][0]*h;
        hw2 = hw*hw;
        kwmax = sqrt(dstarmax2 - hw2);
        
        // kmax = (int) (SI[0][1]*hw + SI[1][1]*kwmax + 1);
        kmax = (int) (SI[1][1]*kwmax + 1);
        kmins[0] = -kmax;
        kmins[2] = h;
        kmin = kmins[limit_k];

        hi = h - hmin;
        p->kmin[hi] = kmin;
        p->kmax[hi] = kmax;
        p->lmin = malloc((kmax-kmin)*sizeof(int));
        p->lmax = malloc((kmax-kmin)*sizeof(int));

        for (k = kmin; k <= kmax; k++) {
            kw = S[0][1]*h + S[1][1]*k;
            kw2 = kw*kw;
            lwmax = sqrt(dstarmax2 - hw2 - kw2);
            
            // lmax = (int) (SIT[0][2]*hw + SIT[1][2]*kw + SIT[2][2]*lwmax + 1);
            lmax = (int) (SI[2][2]*lwmax + 1);
            lmins[2] = h;
            lmins[3] = k;
            lmin = lmins[limit_l];
            nrefln += lmax - lmin + 1; // because <=

            ki = k - kmin;
            p->lmin[hi][ki] = lmin;
            p->lmax[hi][ki] = lmax;
        }
    }

    // allocate memory for reflections stored in Refln structs
    // enum {int i; double f;} Fhkls[nrefln][5];
    p->F = malloc(nrefln*sizeof(Refln));
    p->nrefln = nrefln;
    return p;
}

void free_Reflns(Reflns *p) {
    int hn = p->hmax - p->hmin;
    for (int hi = 0; hi <= hn; hi++) {
        free(p->lmin[hi]);
        free(p->lmax[hi]);
    } 
    free(p->kmin);
    free(p->kmax);
    free(p->F);
    free(p);
}


/*******************************************************************************
*
* Reflection enumeration and reciprocal expansion (to calculate Fhkl)
*
*******************************************************************************/

void calc_Rc(double S[][3], double Rw[][3], double SI[][3], double Rc[][3]) {

    // perform similarity transform to change matrix basis: Rc = S*Rw*SI
    int i, j, k;
    double SR[3][3];
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            SR[i][j] = 0.0;
            for (k = 0; k < 3; k++) {
                SR[i][j] += S[i][k]*Rw[k][j];
            }   
        }   
    }
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            Rc[i][j] = 0.0;
            for (k = 0; k < 3; k++) {
                Rc[i][j] += SR[i][k]*SI[k][j];
            }
        }
    }
}

void map2hkl(double dmin, Atoms *pdb, Grid *rho) {

    // correction for B-smear
    const double sigma = 0.5*3.0; // 0.5*invg;
    const double Q = 100;
    const double Bextra = rho->Bextra;
    const double degrees = 180.0/PI;

    // grid dimensions
    const int na = rho->n3;
    const int nb = rho->n2;
    const int nc = rho->n1;
    const int halfna = na/2+1; // nyquist+1

    // unit cell parameters
    double a = pdb->a;
    double b = pdb->b;
    double c = pdb->c;
    double al = pdb->al;
    double be = pdb->be;
    double ga = pdb->ga;

    // correction for volume/grid points
    const double vox = a*b*c*sqrt(1 - (cos(al)*cos(al)+cos(be)*cos(be)+cos(ga)*cos(ga))
                 + 2*cos(al)*cos(be)*cos(ga))/(1.0*na*nb*nc);

    // allocate space for complex data (if using multiple times only need once)
    // take care of nc/2+1 <-- do we need to change modulo steps below

    // plan optimal FFT (if using multiple times only need once)
    fftw_complex *fftrho = fftw_malloc(nc*nb*na*sizeof(fftw_complex));
    fftw_plan fft = fftw_plan_dft_r2c_3d(nc, nb, na, rho->raw, fftrho,
                                         FFTW_ESTIMATE);

    // carry out fft
    fftw_execute(fft);

    //LOGINFO("rho = %f", rho[49][79][78]);
    //LOGINFO("rho = %f", rho->data[49][79][78]);
    //LOGINFO("Fasu = %f + %fj", fftrho[0][0][0]);

    // scale matrix and its inverse
    double S[3][3], SI[3][3];
    calc_scale(a, b, c, al, be, ga, S);
    calc_invscale(a, b, c, al, be, ga, SI);

    const int nsym = pdb->nsym;
    int j;
    double Rc[nsym][3][3];
    double tw[3];
    double tc[nsym][3];

    // loop over symops to convert to crystallographic/fractional basis 
    for (j = 0; j < nsym; j++) {

        // change basis of rotation matrix by similarity transform Rc = S*Rw*SI
        calc_Rc(S, pdb->sym[j].R, SI, Rc[j]);

        // change basis of translational shift Tc = S*Tw
        memcpy(tw, pdb->sym[j].t, 3*sizeof(double));
        tc[j][0] = S[0][0]*tw[0] + S[0][1]*tw[1] + S[0][2]*tw[2];
        tc[j][1] = S[1][0]*tw[0] + S[1][1]*tw[1] + S[1][2]*tw[2];
        tc[j][2] = S[2][0]*tw[0] + S[2][1]*tw[1] + S[2][2]*tw[2];
    }

    // for loop-invariant code motion of hR
    double hR00[nsym];
    double hR01[nsym];
    double hR02[nsym];
    double hR00_kR10[nsym];
    double hR01_kR11[nsym];
    double hR02_kR12[nsym];

    // for loop-invariant code motion of hT
    double hT0[nsym];
    double hT0_kT1[nsym];

    // laue group - defines limits the unique wedge of reciprocal space
    const int limit_h = laue_limits[pdb->laue][0];
    const int limit_k = laue_limits[pdb->laue][1];
    const int limit_l = laue_limits[pdb->laue][2];

    // radius of reciprocal-space resolution sphere
    const double dstarmax2 = 1.0/(dmin*dmin);

    // unique wedge of reciprocal space (see code in data section above)
    // h = SIT[0][0]*hw + SIT[0][1]*kw + SIT[0][2]*lw
    const int hmax = (int) (SI[0][0]/dmin + 1);
    int hmin = 0; 
    if (limit_h == 0) {
        hmin = -hmax;
    }

    int kmins[3];
    int lmins[4];
    kmins[1] = 0;
    lmins[1] = 0; 

    // local variables
    int h, k, l, hR, kR, lR, kmin, kmax, lmin, lmax, sign;
    double hT, hw, kw, lw, hw2, kw2, lw2, kwmax, lwmax, dstar2;
    complex Fasu, F;
    double A, phi;

    #ifdef TO_FILE
    Reflns *p = create_Reflns(dmin, S);
    int i = 0;
    #endif

    for (h = hmin; h <= hmax; h++) {

        hw = S[0][0]*h; // ST[0][0]*h + ST[0][1]*k + ST[0][2]*l;
        hw2 = hw*hw;
        if (dstarmax2 < hw2) continue;
        kwmax = sqrt(dstarmax2 - hw2);
        // kmax = STI[1][0]*hw + STI[1][1]*kw + STI[1][2]*lw
        kmax = (int) (SI[1][1]*kwmax + 1);

        // unique wedge of reciprocal space (see code in data section above)
        kmins[0] = -kmax;
        kmins[2] = h;
        kmin = kmins[limit_k];

        // scalar promotion
        for (j = 0; j < nsym; j++) {
            hR00[j] = h*Rc[j][0][0];
            hR01[j] = h*Rc[j][0][1];
            hR02[j] = h*Rc[j][0][2];
            hT0[j] = h*tc[j][0];
        }

        for (k = kmin; k <= kmax; k++) {

            kw = S[0][1]*h + S[1][1]*k; // ST[1][0]*h + ST[1][1]*k + ST[1][2]*l;
            kw2 = kw*kw;
            if (dstarmax2 < hw2 + kw2) continue;
            lwmax = sqrt(dstarmax2 - hw2 - kw2);
            // lmax = STI[2][0]*hw + STI[2][1]*kw + STI[2][2]*lw and SI[
            lmax = (int) (SI[2][2]*lwmax + 1); // 

            // unique wedge of reciprocal space (see code in data section above)
            lmins[2] = h;
            lmins[3] = k;
            lmin = lmins[limit_l];

            // scalar promotion
            for (j = 0; j < nsym; j++) {
                hR00_kR10[j] = hR00[j] + k*Rc[j][1][0];
                hR01_kR11[j] = hR01[j] + k*Rc[j][1][1];
                hR02_kR12[j] = hR02[j] + k*Rc[j][1][2];
                hT0_kT1[j] = hT0[j] + k*tc[j][1];
            }

            for (l = lmin; l <= lmax; l++) { // lmin

                // reciprocal space ASU expansion with shifts
                F = 0.0;
                for (j = 0; j < nsym; j++) {

                    // loop over symmetry equivalent reflections hT*R = RT*h
                    hR = round(hR00_kR10[j] + l*Rc[j][2][0]);
                    kR = round(hR01_kR11[j] + l*Rc[j][2][1]);
                    lR = round(hR02_kR12[j] + l*Rc[j][2][2]);

                    // branchless sign function also (hR>=0) - (lR<0)
                    sign = (hR >> 31) | 1;

                    // if sign(hR) = -1 take negative indies (move into grid)
                    hR *= sign;
                    kR *= sign;
                    lR *= sign;

                    // periodicity in na nb and nc (move into grid)
                    kR = (kR + nb) % nb;
                    lR = (lR + nc) % nc;

                    // TODO TEST: reflection about center
                    // hR = (hR < na/2) ? hR : na/2 - hR;

                    // memor access pattern un-important due to symop loop
                    // see http://www.fftw.org/doc/Dynamic-Arrays-in-C.html
                    Fasu = fftrho[hR + (na/2+1)*(kR + nb*lR)];

                    // if sign(hR) = +1 - take complex conjugates to get iFFT
                    // if sign(hR) = -1 - negated indicies so just FFT result
                    Fasu = creal(Fasu) - sign*cimag(Fasu)*I;

                    // correct for translational shifts
                    hT = hT0_kT1[j] + l*tc[j][2]; // h*tc[0]+k*tc[1]+l*tc[2];
                    hT -= round(hT); // hT%1
                    F += Fasu * cexp(2*PI*I*hT);
                }

                // lw = ST[2][0]*h + ST[2][1]*k + ST[2][2]*l;
                lw = S[0][2]*h + S[1][2]*k + S[2][2]*l;
                // reciprocal resolution squared
                dstar2 = hw2 + kw2 + lw*lw;

                // apply B-smear and voume per grid point corrections
                F *= vox * exp(Bextra*dstar2*0.25);
                // convert to amplitude and phase (cartesian to polar coord)
                A = cabs(F);
                phi = carg(F)*degrees;
                
                #ifdef TO_FILE
                p->F[i].h = h;
                p->F[i].k = k;
                p->F[i].l = l;
                p->F[i].A = A;
                p->F[i].phi = phi;
                i++;
                #else
                printf("[%5d %5d %5d] [%10.3f %10.3f]\n", h, k, l, A, phi);
                #endif
            }
        }
    }

    fftw_free(fftrho);

    #ifdef TO_FILE
    FILE *fp = fopen(OUTPUT, "w");
    ASSERT_EXIT(fp != NULL, "Error opening out file");
    for (i = 0; i < p->nrefln; i++) {
        fprintf(fp, "[%5d %5d %5d] [%10.3f %10.3f]\n", p->F[i].h, p->F[i].k,
                p->F[i].l, p->F[i].A, p->F[i].phi);
    }
    fclose(fp);
    free_Reflns(p);
    #endif
}

/*******************************************************************************
*
* Simulate diffraction pattern
*
*******************************************************************************/

/*******************************************************************************
*
* ALTERNATIVE - direct summation - for checking and speed tests
*
*******************************************************************************/

void pdb2hkl_direct(int h, int k, int l, Atoms *pdb) {

    double S[3][3];
    calc_scale(pdb->a, pdb->b, pdb->c, pdb->al, pdb->be, pdb->ga, S);

    double hw = S[0][0]*h;
    double kw = S[0][1]*h + S[1][1]*k;
    double lw = S[0][2]*h + S[1][2]*k + S[2][2]*l;
    double stol2 = (hw*hw + kw*kw + lw*lw)*0.25;

    int e, ei;
    double n, B, xc, yc, zc, f0[pdb->nelems];
    complex F = 0;

    for (int ei = 0; ei < pdb->nelems; ei++) {
        e = pdb->new2old[ei];
        f0[ei] = CCM[e];
        for (int j = 0; j < 4; j++) {
            f0[ei] += ACM[e][j]*exp(-BCM[e][j]*stol2);
        }
    }

    for (int i = 0; i < pdb->natoms; i++) {
        B = pdb->B[i];
        n = pdb->n[i];
        ei = pdb->newe[i];
        xc = S[0][0]*pdb->x[i] + S[0][1]*pdb->y[i] + S[0][2]*pdb->z[i];
        yc = S[1][1]*pdb->y[i] + S[1][2]*pdb->z[i];
        zc = S[2][2]*pdb->z[i];
        F += n * f0[ei] * exp(-B*stol2) * cexp(I*2*PI*(h*xc + k*yc + l*zc));
    }

    double A = cabs(F);
    double phi = carg(F)*180.0/PI;
    printf("[%5d %5d %5d] [%10.3f %10.3f]\n", h, k, l, A, phi);
}

void pdb2hkl_ndirect(double dmin, Atoms *pdb) {

    #ifdef TO_FILE
    FILE *fp = fopen(OUTPUT, "w");
    #endif

    double xcs[pdb->natoms], ycs[pdb->natoms], zcs[pdb->natoms];
    // double hxcs[pdb->natoms], kycs[pdb->natoms];
    double Bs[pdb->natoms], ns[pdb->natoms], f0s[pdb->nelems];
    int i, e, ei, es[pdb->natoms];
    double xc, yc, zc, n, B;
    int h, k, l, hmin, hmax, kmin, kmax, lmin, lmax, limit_h, limit_k, limit_l;
    int kmins[3], lmins[4];
    double hw2, kw2, lw2, kwmax, lwmax;
    double stol2, A, phi;
    complex F, tauj = I*2*PI;

    double dstarmax2 = 1.0/(dmin*dmin);

    double S[3][3], SI[3][3];
    calc_scale(pdb->a, pdb->b, pdb->c, pdb->al, pdb->be, pdb->ga, S);
    calc_invscale(pdb->a, pdb->b, pdb->c, pdb->al, pdb->be, pdb->ga, SI);

    // crystallographic coordinates
    for (i = 0; i < pdb->natoms; i++) {
        es[i] = pdb->newe[i];
        Bs[i] = pdb->B[i];
        ns[i] = pdb->n[i];
        xcs[i] = S[0][0]*pdb->x[i] + S[0][1]*pdb->y[i] + S[0][2]*pdb->z[i];
        ycs[i] = S[1][1]*pdb->y[i] + S[1][2]*pdb->z[i];
        zcs[i] = S[2][2]*pdb->z[i];
    }

    // unique wedge of reciprocal space
    limit_h = laue_limits[pdb->laue][0];
    limit_k = laue_limits[pdb->laue][1];
    limit_l = laue_limits[pdb->laue][2];
    hmax = (int) (SI[0][0]/dmin + 1);
    hmin = (limit_h == 0) ? -hmax : 0;
    kmins[1] = 0;
    lmins[1] = 0;

    // loop over hkl
    for (h = hmin; h <= hmax; h++) {
        hw2 = S[0][0]*h;
        hw2 *= hw2;
        if (dstarmax2 < hw2) continue;
        kwmax = sqrt(dstarmax2 - hw2);
        kmax = (int) (SI[1][1]*kwmax + 1);
        kmins[0] = -kmax;
        kmins[2] = h;
        kmin = kmins[limit_k];
        //for (i = 0; i < pdb->natoms; i++) hxcs[i] = h*xcs[i];

        for (k = kmin; k <= kmax; k++) {
            kw2 = S[0][1]*h + S[1][1]*k;
            kw2 *= kw2;
            if (dstarmax2 < hw2 + kw2) continue;
            lwmax = sqrt(dstarmax2 - hw2 - kw2);
            lmax = (int) (SI[2][2]*lwmax + 1);
            lmins[2] = h;
            lmins[3] = k;
            lmin = lmins[limit_l];
            //for (i = 0; i < pdb->natoms; i++) kycs[i] = k*ycs[i];

            for (l = lmin; l <= lmax; l++) {
                lw2 = S[0][2]*h + S[1][2]*k + S[2][2]*l;
                lw2 *= lw2;
                stol2 = (hw2 + kw2 + lw2)*0.25;

                // loop over elements
                for (ei = 0; ei < pdb->nelems; ei++) {
                    e = pdb->new2old[ei];
                    f0s[ei] = CCM[e];
                    for (int j = 0; j < 4; j++) {
                        f0s[ei] += ACM[e][j]*exp(-BCM[e][j]*stol2);
                    }
                }

                // loop over atoms cexp is expensive could do lookup
                F = 0.0;
                for (i = 0; i < pdb->natoms; i++) {
                    F += pdb->n[i] * f0s[pdb->newe[i]] *
                         cexp(-pdb->B[i]*stol2 + tauj*(h*xcs[i] + k*ycs[i] + l*zcs[i]));
                }

                A = cabs(F);
                phi = carg(F)*180/PI;
                #ifdef TO_FILE
                fprintf(fp, "[%5d %5d %5d] [%10.3f %10.3f]\n", h, k, l, A, phi);
                #else
                printf("[%5d %5d %5d] [%10.3f %10.3f]\n", h, k, l, A, phi);
                #endif
            }
        }
    }

    #ifdef TO_FILE
    fclose(fp);
    #endif
}

/*******************************************************************************
*
* main function called when application executes
*
*******************************************************************************/

int main(int argc, char **argv) {

    ASSERT_EXIT(argc >= 5, "USAGE: %s file.pdb <invg> <rhocut> <dmin> <bbase>",
                argv[0]);

    double invg = atof(argv[2]); // 3.0 or 2.0 or 1.0
    double rhocut = atof(argv[3]); // 0.01 or 0.005
    double dmin = atof(argv[4]); // 17.0
    double Bextra;

    ASSERT_EXIT(invg >= 1, "Error: inverse g < 1");
    ASSERT_EXIT(dmin >= 1, "Error: resolution < 1 Angstrom"); // see map2hkl TODO
    ASSERT_EXIT(rhocut <= 0.05, "Error: rho cutoff > 0.05");

    LOGINFO("file = %s", argv[1]);
    LOGINFO("invg = %f", invg);
    LOGINFO("rhocut = %f", rhocut);
    LOGINFO("dmin = %f", dmin);

    Atoms *pdb = parse_pdb(argv[1]);
    // LOGINFO("spg = %s", pdb->spg);

    Bextra = (argc == 6) ? atof(argv[5]) : calc_Bextra(100, 1.0, 3.0, pdb->Bmin);
    LOGINFO("Bextra = %f", Bextra);

    Grid *rho = pdb2map(1.0, invg, rhocut, Bextra, pdb);
    map2hkl(dmin, pdb, rho);

    // pdb2hkl_ndirect(dmin, pdb);
    // pdb2hkl_direct(3, 2, 2, pdb);
    // pdb2hkl_direct(3, 3, 0, pdb);

    free_Atoms(pdb);
    free_Grid(rho);
}
