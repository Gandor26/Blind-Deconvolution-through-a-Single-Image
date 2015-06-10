#ifndef COMPRESSEDSENSE_H
#define COMPRESSEDSENSE_H
#include <core/core_c.h>
#include <core/types_c.h>

#ifndef DOUBLE_MAX
#define DOUBLE_MAX 2147483647.0
#endif // DOUBLE_MAX
#define lambda 1.0

/**
    an optimization system in such a form:

        min ||Ax - y||_2^2 + lambda*||x||_1

    with possible linear inequality constraints, and in this model, lambda = 1
*/

class compressedSense
{
    public:
        compressedSense(int kernel_count, int img_count);
        virtual ~compressedSense();
        void solveOptm();
        //void setParameters();
        void Mat2Vector(CvMat *L, CvMat *I, CvSize imgSize, CvSize kernelSize);
        void Vector2Mat(CvMat *f);
    public:
        void PCG(CvMat* A, CvMat* b, CvMat* x, CvMat* Pinv, double tol, int maxiter);
    public:
        void calcNewtonDirection(CvMat *dx, CvMat *du, CvMat *gradPhi_t);
        void backtrackingLineSearch();
        double phi_t(CvMat *x, CvMat *u, CvMat *z);

        int kernelCount, imgCount;      //the pixel numbers of kernel and image, respectively
        CvMat *A, *x, *y;               //elements in the optimization problem
        CvMat *u;                       //constraint barriers
        CvMat *z;                       //z = Ax-y
        CvMat *v;                       //dual variance
        CvMat *dx, *du;                 //Newton direction of phi_t
        CvMat *gradPhi_t;               //grad

        double eps;                     //tolerance of iteration error
        double t;                       //control weight
        double primal_value;
        double dual_value;
        double eita;                    //gap between primal objective value and lagrange dual value
        double miu;                     //¦Ì is the factor in updating t, default(2)
        double s_min;                   //s_min is a threshold in determining the next t, default(0.5)
        double alpha, beta;             //¦Á, ¦Â are paramters in backtracking line search
        double s;                       //scaling const of dual variance v
};

#endif // COMPRESSEDSENSE_H
