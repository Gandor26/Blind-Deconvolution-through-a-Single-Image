#include "compressedSense.h"
#include <iostream>
using namespace std;

compressedSense::compressedSense(int kernel_count, int img_count)
{
    kernelCount = kernel_count;
    imgCount = img_count;

    A = cvCreateMat(imgCount, kernelCount, CV_64FC1);
    x = cvCreateMat(kernelCount, 1, CV_64FC1);
    y = cvCreateMat(imgCount, 1, CV_64FC1);
    u = cvCreateMat(kernelCount, 1, CV_64FC1);
    z = cvCreateMat(imgCount, 1, CV_64FC1);
    v = cvCreateMat(imgCount, 1, CV_64FC1);
    dx = cvCreateMat(kernelCount, 1, CV_64FC1);
    du = cvCreateMat(kernelCount, 1, CV_64FC1);
    gradPhi_t = cvCreateMat(2*kernelCount, 1, CV_64FC1);

    primal_value = DOUBLE_MAX;
    dual_value = -DOUBLE_MAX;
    t = 1;
    alpha = 0.01;
    beta = 0.5;
    miu  = 2;
    s_min = 0.5;
    s = DOUBLE_MAX;
    cvSetZero(x);
    cvSet(u, cvScalarAll(1), NULL);
}

compressedSense::~compressedSense()
{
    cvReleaseMat(&u);
    cvReleaseMat(&z);
    cvReleaseMat(&v);
    cvReleaseMat(&dx);
    cvReleaseMat(&du);
    cvReleaseMat(&gradPhi_t);
}

void compressedSense::Mat2Vector(CvMat *L, CvMat* I, CvSize imgSize, CvSize kernelSize)
{
    for (int h = 0; h < imgSize.height; h++)
    {
        for (int w = 0; w < imgSize.width; w++)
        {
            int row = h*imgSize.width + w;
            cvmSet(y, row, 0, cvmGet(I, h, w));

            for (int kh = 0; kh < kernelSize.height; kh++)
            {
                for (int kw = 0; kw < kernelSize.width; kw++)
                {
                    int col = kh*kernelSize.width + kw;
                    int py = h + kh - kernelSize.height/2;
                    int px = w + kw - kernelSize.width/2;
                    if (py < 0 || py >= L->rows || px < 0 || px >= L->cols)
                        continue;
                    else
                        cvmSet(A, row, col, cvmGet(L, py, px));
                }
            }
        }
    }
    cout << "Vectorized done!" << endl;
}

void compressedSense::Vector2Mat(CvMat *f)
{
    for (int h = 0; h < f->rows; h++)
    {
        for (int w = 0; w < f->cols; w++)
        {
            int row = h*f->cols + w;
            cvmSet(f, h, w, cvmGet(x, row, 0));
        }
    }
    cout << "Matriclized done!" << endl;
}

void compressedSense::MulA(CvMat* A, CvMat* d1, CvMat* d2, CvMat* x, CvMat* dst)
{
    int dim = A->rows;
    if (x->rows != dst->rows || x->rows != dim * 2)
    {
        cerr << "Matrix Size Error!" << endl;
        return;
    }
    CvMat* x1 = cvCreateMat(dim, 1, CV_64FC1);
    CvMat* x2 = cvCreateMat(dim, 1, CV_64FC1);
    CvMat* dst1 = cvCreateMat(dim, 1, CV_64FC1);
    CvMat* dst2 = cvCreateMat(dim, 1, CV_64FC1);
    CvMat* tmp = cvCreateMat(dim, 1, CV_64FC1);
    for (int i = 0; i < dim; i++)
    {
        cvmSet(x1, i, 0, cvmGet(x, i, 0));
        cvmSet(x2, i, 0, cvmGet(x, i+dim, 0));
    }
    cvGEMM(A, x1, 1.0, NULL, 0.0, dst1);
    cvMul(d1, x1, tmp, 1.0);
    cvAdd(dst1, tmp, dst1);
    cvMul(d2, x2, tmp, 1.0);
    cvAdd(dst1, tmp, dst1);
    cvMul(d2, x1, dst2, 1.0);
    cvMul(d1, x2, tmp, 1.0);
    cvAdd(dst2, tmp, dst2);
    for (int i = 0; i < dim; i++)
    {
        cvmSet(dst, i, 0, cvmGet(dst1, i, 0));
        cvmSet(dst, i+dim, 0, cvmGet(dst2, i, 0));
    }
    cvReleaseMat(&x1);
    cvReleaseMat(&x2);
    cvReleaseMat(&dst1);
    cvReleaseMat(&dst2);
    cvReleaseMat(&tmp);
}

void compressedSense::MulPinv(CvMat* p1, CvMat* p2, CvMat* p3, CvMat* x, CvMat* dst)
{
    int dim = p1->rows;
    if (x->rows != dst->rows || x->rows != dim * 2 || p1->rows != p2->rows || p1->rows != p2->rows)
    {
        cerr << "Matrix Size Error!" << endl;
        return;
    }
    CvMat* x1 = cvCreateMat(dim, 1, CV_64FC1);
    CvMat* x2 = cvCreateMat(dim, 1, CV_64FC1);
    CvMat* dst1 = cvCreateMat(dim, 1, CV_64FC1);
    CvMat* dst2 = cvCreateMat(dim, 1, CV_64FC1);
    CvMat* tmp = cvCreateMat(dim, 1, CV_64FC1);
    for (int i = 0; i < dim; i++)
    {
        cvmSet(x1, i, 0, cvmGet(x, i, 0));
        cvmSet(x2, i, 0, cvmGet(x, i+dim, 0));
    }
    cvMul(p1, x1, dst1, 1.0);
    cvMul(p2, x2, tmp, 1.0);
    cvSub(dst1, tmp, dst1);
    cvMul(p2, x1, dst2, 1.0);
    cvMul(p3, x2, tmp, 1.0);
    cvSub(tmp, dst2, dst2);
    for (int i = 0; i < dim; i++)
    {
        cvmSet(dst, i, 0, cvmGet(dst1, i, 0));
        cvmSet(dst, i+dim, 0, cvmGet(dst2, i, 0));
    }
    cvReleaseMat(&x1);
    cvReleaseMat(&x2);
    cvReleaseMat(&dst1);
    cvReleaseMat(&dst2);
    cvReleaseMat(&tmp);
}

void compressedSense::PCG(CvMat* A, CvMat* b, CvMat* x, CvMat* d1, CvMat* d2,
                          CvMat* p1, CvMat* p2, CvMat* p3, double tol, int maxiter)
{
    int dim = x->rows;
    int iter = 0;
    bool flag = false;

    cvSetZero(x);
    CvMat *r = cvCloneMat(b);
    CvMat *z = cvCreateMat(dim, 1, CV_64FC1);
//    if (Pinv == NULL)
//    {
//        Pinv = cvCreateMat(dim, 1, CV_64FC1);
//        cvSetIdentity(Pinv);
//    }
    MulPinv(p1, p2, p3, r, z);
    CvMat *p = cvCloneMat(z);
    CvMat *Ap = cvCreateMat(dim, 1, CV_64FC1);

    while (1)
    {
        iter++;
        if (iter > maxiter) break;
        MulA(A, d1, d2, p, Ap);
        double rou = cvDotProduct(r, z);
        double alpha = rou / cvDotProduct(p, Ap);
        cvScaleAdd(p, cvScalarAll(alpha), x, x);
        cvScaleAdd(Ap, cvScalarAll(-alpha), r, r);
        MulPinv(p1, p2, p3, r, z);
        double beta = cvDotProduct(r, z) / rou;
        cvScaleAdd(p, cvScalarAll(beta), z, p);
        if (cvNorm(r, NULL, CV_L2, NULL) < eps)
        {
            flag = true;
            break;
        }
    }
    if (flag) cout << "Absolute tolerance reached!" << endl;
    else if (iter > maxiter) cout << "Max iteration exceeded!" << endl;
    cvReleaseMat(&r);
    cvReleaseMat(&z);
    cvReleaseMat(&p);
    cvReleaseMat(&Ap);
}

void compressedSense::calcNewtonDirection(CvMat *dx, CvMat *du, CvMat *gradPhi_t)
{
    CvMat *tmpN = cvCreateMat(kernelCount, 1, CV_64FC1);
    CvMat *tmpNN = cvCreateMat(kernelCount, kernelCount, CV_64FC1);
    CvMat *q1 = cvCreateMat(kernelCount, 1, CV_64FC1);
    CvMat *q2 = cvCreateMat(kernelCount, 1, CV_64FC1);
    CvMat *d1 = cvCreateMat(kernelCount, 1, CV_64FC1);
    CvMat *d2 = cvCreateMat(kernelCount, 1, CV_64FC1);

    /*      Phi_t(x, u) = t|Ax-y|^2 + t*lambda*|u|_1 - |log(u+x)|_1 - |log(u-x)|_1      */

    for (int i = 0; i < kernelCount; i++)
    {
        cvmSet(q1, i, 0, 1.0/(cvmGet(u, i, 0) + cvmGet(x, i, 0)));
        cvmSet(q2, i, 0, 1.0/(cvmGet(u, i, 0) - cvmGet(x, i, 0)));
        cvmSet(d1, i, 0, (cvmGet(q1, i, 0)*cvmGet(q1, i, 0) + cvmGet(q2, i, 0)*cvmGet(q2, i, 0))/t);
        cvmSet(d2, i, 0, (cvmGet(q1, i, 0)*cvmGet(q1, i, 0) - cvmGet(q2, i, 0)*cvmGet(q2, i, 0))/t);
    }

    //CvMat *hessPhi_t = cvCreateMat(2*kernelCount, 2*kernelCount, CV_64FC1);
    CvMat *dxdu = cvCreateMat(2*kernelCount, 1, CV_64FC1);
    CvMat *p1 = cvCreateMat(kernelCount, 1, CV_64FC1);
    CvMat *p2 = cvCreateMat(kernelCount, 1, CV_64FC1);
    CvMat *p3 = cvCreateMat(kernelCount, 1, CV_64FC1);
    //CvMat *PrecondInv = cvCreateMat(2*kernelCount, 2*kernelCount, CV_64FC1);
    //cvSetZero(PrecondInv);
    cvGEMM(A, z, 2.0, NULL, 0.0, tmpN, CV_GEMM_A_T);
    cvGEMM(A, A, 2.0, NULL, 0.0, tmpNN, CV_GEMM_A_T);       //tmpNN = 2A'A = grad^2(||Ax-y||^2);

    for (int i = 0; i < kernelCount; i++)
    {
        cvmSet(gradPhi_t, i, 0, cvmGet(tmpN, i, 0) - (cvmGet(q1, i, 0) - cvmGet(q2, i, 0)) / t);    //gradPhi_t(1:N) = 2A'*z-(1./(u+x)-1./(u-x))/t
        cvmSet(gradPhi_t, i+kernelCount, 0, lambda - (cvmGet(q1, i, 0) + cvmGet(q2, i, 0)) / t);    //gradPhi_t(N+1:2N) = lambda - (1./(u+x)+1./(u-x))/t
        double prb = cvmGet(d1, i, 0) + cvmGet(tmpNN, i, i);
        double prs = prb * cvmGet(d1, i, 0) - cvmGet(d2, i, 0)*cvmGet(d2, i, 0);
        cvmSet(p1, i, 0, cvmGet(d1, i, 0) / prs);
        cvmSet(p2, i, 0, cvmGet(d2, i, 0) / prs);
        cvmSet(p3, i, 0, prb / prs);

//        for (int j = 0; j < kernelCount; j++)
//        {
//            double x1 = (j == i)?cvmGet(d1, i, 0):0;
//            double x2 = (j == i)?cvmGet(d2, i, 0):0;
//            cvmSet(hessPhi_t, i, j, cvmGet(tmpNN, i, j) + x1);
//            cvmSet(hessPhi_t, i+kernelCount, j, x2);
//            cvmSet(hessPhi_t, i, j+kernelCount, x2);
//            cvmSet(hessPhi_t, i+kernelCount, j+kernelCount, x1);
//            if (j == i)
//            {
//                cvmSet(PrecondInv, i, i, x1 / prs);
//                cvmSet(PrecondInv, i+kernelCount, i, -x2 / prs);
//                cvmSet(PrecondInv, i, i+kernelCount, -x2 / prs);
//                cvmSet(PrecondInv, i+kernelCount, i+kernelCount, prb / prs);
//            }
//        }
    }

//    for (int i = 0; i < 2*kernelCount; i++)
//    {
//        for(int j = 0; j < 2*kernelCount; j++)
//        {
//            cout << cvmGet(hessPhi_t, i, j)<<' ';
//        }
//        cout << endl;
//    }
//    for(int i = 0; i < 2*kernelCount; i++)
//        cout << -cvmGet(gradPhi_t, i, 0) << ' ';

    double normg = cvNorm(gradPhi_t, NULL, CV_L2, NULL);
    double pcgtol = min(1e-1, 1e-3*eita/min(1.0, normg));
    cvConvertScale(gradPhi_t, gradPhi_t, -1.0, 0.0);
    PCG(tmpNN, gradPhi_t, dxdu, d1, d2, p1, p2, p3, pcgtol, 20);
    cvConvertScale(gradPhi_t, gradPhi_t, -1.0, 0.0);

    for (int i = 0; i < kernelCount; i++)
    {
        cvmSet(dx, i, 0, cvmGet(dxdu, i, 0));
        cvmSet(du, i, 0, cvmGet(dxdu, i+kernelCount, 0));
    }

    cvReleaseMat(&tmpN);
    cvReleaseMat(&tmpNN);
    cvReleaseMat(&q1);
    cvReleaseMat(&q2);
    cvReleaseMat(&d1);
    cvReleaseMat(&d2);
    cvReleaseMat(&p1);
    cvReleaseMat(&p2);
    cvReleaseMat(&p3);
    //cvReleaseMat(&hessPhi_t);
    cvReleaseMat(&dxdu);
    //cvReleaseMat(&PrecondInv);
}

double compressedSense::phi_t(CvMat* x, CvMat* u, CvMat* z)
{
    CvMat *q1 = cvCreateMat(kernelCount, 1, CV_64FC1);
    CvMat *q2 = cvCreateMat(kernelCount, 1, CV_64FC1);
    CvScalar s1, s2, su;
    cvAdd(u, x, q1, NULL);      cvSub(u, x, q2, NULL);
    cvLog(q1, q1);              cvLog(q2, q2);
    s1 = cvSum(q1);             s2 = cvSum(q2);
    su = cvSum(u);

    cvReleaseMat(&q1);      cvReleaseMat(&q2);
    return cvDotProduct(z, z) + lambda*su.val[0] - (s1.val[0]+s2.val[0]) / t;
}

void compressedSense::backtrackingLineSearch()
{
    int iter = 0;
    double phi = phi_t(x, u, z), newphi;
    s = 1.0;
    CvMat *dxu = cvCreateMat(2*kernelCount, 1, CV_64FC1);
    CvMat *newx = cvCreateMat(kernelCount, 1, CV_64FC1);
    CvMat *newu = cvCreateMat(kernelCount, 1, CV_64FC1);
    CvMat *newz = cvCreateMat(imgCount, 1, CV_64FC1);
    for (int i = 0; i < kernelCount; i++)
    {
        cvmSet(dxu, i, 0, cvmGet(dx, i, 0));
        cvmSet(dxu, i+kernelCount, 0, cvmGet(du, i, 0));
    }
    double gdx = cvDotProduct(gradPhi_t, dxu);
    while(1)
    {
        iter++;
        if (iter > 100) break;
        cvScaleAdd(dx, cvScalarAll(s), x, newx);
        cvScaleAdd(du, cvScalarAll(s), u, newu);
        bool positive = true;
        for (int i = 0; i < kernelCount; i++)
        {
            double u_i = cvmGet(newu, i, 0);
            double x_i = cvmGet(newx, i, 0);
            if (u_i + x_i <= 0 || u_i-x_i <= 0)
            {
                positive = false;
                break;
            }
        }
        if (positive)
        {

            cvGEMM(A, newx, 1.0, y, -1.0, newz, 0);
            newphi = phi_t(newx, newu, newz);
            if (newphi - phi <= alpha*s*gdx) break;
        }
        s *= beta;
    }
    cvConvert(newx, x);
    cvConvert(newu, u);
    //cvConvert(newz, z);

    cvReleaseMat(&newx);
    cvReleaseMat(&newu);
    cvReleaseMat(&newz);
    cvReleaseMat(&dxu);
}

void compressedSense::solveOptm()
{
    /**
        A is M*N matrix, M = imgCount, N = kernelCount
    */
    int iter_count = 0;
    CvMat *tmpN = cvCreateMat(kernelCount, 1, CV_64FC1);
    do
    {
        /*          update v            */

        cvGEMM(A, x, 1.0, y, -1.0, z, 0);                           //z = Ax-y
        cvConvertScale(z, v, 2.0, 0.0);
        cvGEMM(A, v, 1.0, NULL, 0.0, tmpN, CV_GEMM_A_T);
        double Av_inf = cvNorm(tmpN, NULL, CV_C, NULL);
        if (Av_inf > lambda)
        {
            cvConvertScale(v, v, lambda/Av_inf, 0.0);               //v = 2*min{1, 1/||A'v||_inf}*z
        }

        /*          calculate primal and dual           */

        primal_value = cvDotProduct(z, z) + lambda*cvNorm(x, NULL, CV_L1, NULL);
        double dual = -cvDotProduct(v, v)*0.25 - cvDotProduct(v, y);
        dual_value = (dual > dual_value)?dual:dual_value;
        eita = primal_value - dual_value;

        /*          stop criterion          */

        cout << "primal = " << primal_value << ", dual = " << dual_value << ", gap = " << eita << endl;
        if (eita / dual <= eps) break;

        /*           update t           */

        if (s >= s_min)
        {
            t = max(miu*min(2*(double)kernelCount/eita, t), t);
        }
        cout << "t = " << t << endl;

        /*          compute approximate Newton Direction         */

        calcNewtonDirection(dx, du, gradPhi_t);

        /*          backtracking line search         */

        backtrackingLineSearch();
        cout << "s = " << s << endl;

        iter_count++;
    }while (iter_count <= 20);

    cvReleaseMat(&tmpN);
}
