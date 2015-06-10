#include "deConvolution.h"
#include "compressedSense.h"
#include <highgui.h>
#include <complex>

using namespace std;

deConvolution::deConvolution(IplImage *blurredImg, CvSize kerSize)
{
    src = blurredImg;
    width   = blurredImg->width;
    height  = blurredImg->height;
    imgSize = cvGetSize(blurredImg);

    kernelWidth     = kerSize.width;
    kernelHeight    = kerSize.height;
    kernelSize      = kerSize;

    blurred         = cvCreateMat(height, width, CV_64FC1);
    original        = cvCreateMat(height, width, CV_64FC1);
    prev_original   = cvCreateMat(height, width, CV_64FC1);
    kernel          = cvCreateMat(kernelHeight, kernelWidth, CV_64FC1);
    prev_kernel     = cvCreateMat(kernelHeight, kernelWidth, CV_64FC1);

    psi_x           = cvCreateMat(height, width, CV_64FC1);
    prev_psi_x      = cvCreateMat(height, width, CV_64FC1);
    psi_y           = cvCreateMat(height, width, CV_64FC1);
    prev_psi_y      = cvCreateMat(height, width, CV_64FC1);
    I_grad_x        = cvCreateMat(height, width, CV_64FC1);
    I_grad_y        = cvCreateMat(height, width, CV_64FC1);
    L_grad_x        = cvCreateMat(height, width, CV_64FC1);
    L_grad_y        = cvCreateMat(height, width, CV_64FC1);
    F_delta         = cvCreateMat(height, width, CV_64FC1);

    for (int i = 0; i < 2; i++)
    {
        F_blurred[i]    = cvCreateMat(height, width, CV_64FC1);
        F_grad_x[i]     = cvCreateMat(height, width, CV_64FC1);
        F_grad_y[i]     = cvCreateMat(height, width, CV_64FC1);
        F_psi_x[i]      = cvCreateMat(height, width, CV_64FC1);
        F_psi_y[i]      = cvCreateMat(height, width, CV_64FC1);
        F_kernel[i]     = cvCreateMat(height, width, CV_64FC1);
    }

    SmoothRegion = cvCreateMat(height, width, CV_64FC1);
    FFTRegion    = cvCreateMat(height, width, CV_64FC2);

    cvConvertScale(blurredImg, blurred, 1.0/256.0, 0.0);      //scale values in [0, 1]

    getGradient(blurred, I_grad_x, kXdirection);
    getGradient(blurred, I_grad_y, kYdirection);

    cvSetZero(F_grad_x[0]);
    cvmSet(F_grad_x[0], 0, 0, 1.0);    cvmSet(F_grad_x[0], 0, 1, -1.0);
    startFFT(F_grad_x[0], F_grad_x[1], FFTRegion);
    cvSetZero(F_grad_y[0]);
    cvmSet(F_grad_y[0], 0, 0, 1.0);    cvmSet(F_grad_y[0], 1, 0, -1.0);
    startFFT(F_grad_y[0], F_grad_y[1], FFTRegion);
    cvConvert(blurred, F_blurred[0]);
    startFFT(F_blurred[0], F_blurred[1], FFTRegion);

    CvMat *gradReal = cvCreateMat(height, width, CV_64FC1);
    CvMat *gradImag = cvCreateMat(height, width, CV_64FC1);
    double w = weightInit;
    cvSetZero(F_delta);

    /*      compute delta in updating L       */
    cvSetZero(gradReal);
    cvmSet(gradReal, 0, 0, 1.0);
    startFFT(gradReal, gradImag, FFTRegion);
    for (int i = 0; i < gradReal->rows; i++)
    {
        for(int j = 0; j < gradReal->cols; j++)
        {
            complex<double> par(cvmGet(gradReal, i, j), cvmGet(gradImag, i, j));
            cvmSet(F_delta, i, j, cvmGet(F_delta, i, j)+norm(par)*w);
        }
    }

    w /= 2.0;                               //omage = 50/(2^q), q = |grad|
    cvSetZero(gradReal);
    cvmSet(gradReal, 0, 0, 1.0);  cvmSet(gradReal, 0, 1, -1.0);
    startFFT(gradReal, gradImag, FFTRegion);
    for (int i = 0; i < gradReal->rows; i++)
    {
        for(int j = 0; j < gradReal->cols; j++)
        {
            complex<double> par(cvmGet(gradReal, i, j), cvmGet(gradImag, i, j));
            cvmSet(F_delta, i, j, cvmGet(F_delta, i, j)+norm(par)*w);
        }
    }
    cvSetZero(gradReal);
    cvmSet(gradReal, 0, 0, 1.0);  cvmSet(gradReal, 1, 0, -1.0);
    startFFT(gradReal, gradImag, FFTRegion);
    for (int i = 0; i < gradReal->rows; i++)
    {
        for(int j = 0; j < gradReal->cols; j++)
        {
            complex<double> par(cvmGet(gradReal, i, j), cvmGet(gradImag, i, j));
            cvmSet(F_delta, i, j, cvmGet(F_delta, i, j)+norm(par)*w);
        }
    }

    w /= 2.0;
    cvSetZero(gradReal);
    cvmSet(gradReal, 0, 0, -1.0);
    cvmSet(gradReal, 0, 1, 2.0);
    cvmSet(gradReal, 0, 2, -1.0);
    startFFT(gradReal, gradImag, FFTRegion);
    for (int i = 0; i < gradReal->rows; i++)
    {
        for(int j = 0; j < gradReal->cols; j++)
        {
            complex<double> par(cvmGet(gradReal, i, j), cvmGet(gradImag, i, j));
            cvmSet(F_delta, i, j, cvmGet(F_delta, i, j)+norm(par)*w);
        }
    }
    cvSetZero(gradReal);
    cvmSet(gradReal, 0, 0, -1.0);
    cvmSet(gradReal, 1, 0, 2.0);
    cvmSet(gradReal, 2, 0, -1.0);
    startFFT(gradReal, gradImag, FFTRegion);
    for (int i = 0; i < gradReal->rows; i++)
    {
        for(int j = 0; j < gradReal->cols; j++)
        {
            complex<double> par(cvmGet(gradReal, i, j), cvmGet(gradImag, i, j));
            cvmSet(F_delta, i, j, cvmGet(F_delta, i, j)+norm(par)*w);
        }
    }
    cvSetZero(gradReal);
    cvmSet(gradReal, 0, 0, -1.0);
    cvmSet(gradReal, 1, 0, 1.0);
    cvmSet(gradReal, 0, 1, 1.0);
    cvmSet(gradReal, 1, 1, -1.0);
    startFFT(gradReal, gradImag, FFTRegion);
    for (int i = 0; i < gradReal->rows; i++)
    {
        for(int j = 0; j < gradReal->cols; j++)
        {
            complex<double> par(cvmGet(gradReal, i, j), cvmGet(gradImag, i, j));
            cvmSet(F_delta, i, j, cvmGet(F_delta, i, j)+norm(par)*w);
        }
    }

    cvReleaseMat(&gradReal);    cvReleaseMat(&gradImag);

    lambdas[0] = lambda_1;
    lambdas[1] = lambda_2;
    gamma = gammaInit;

    cvSetZero(kernel);
    for (int i = 0; i < kernelHeight; i++)
    {
        for (int j = 0; j < kernelWidth; j++)
        {
            if (i==j) cvmSet(kernel, i, j, 1);
        }
    }
}

deConvolution::~deConvolution()
{

}

IplImage *deConvolution::findSmooth(int WindowSize, double threshold)
{
    int offset = WindowSize >> 1;
    CvScalar sd, avg;
    IplImage *smoothRegion = cvCreateImage(cvSize(src->width, src->height), src->depth, 1);
    BWImage image(smoothRegion);
    for (int y = 0; y < src->height - WindowSize; y++)
    {
        for (int x = 0; x < src->width - WindowSize; x++)
        {
            cvSetImageROI(src, cvRect(x, y, WindowSize, WindowSize));
            cvAvgSdv(src, &avg, &sd);
            if (sd.val[0] + sd.val[1] + sd.val[2] >= threshold)
            {
                image[y+offset][x+offset] = 0;
                cvmSet(SmoothRegion, y, x, 0);
            }
            else
            {
                image[y+offset][x+offset] = 255;
                cvmSet(SmoothRegion, y, x, 1);
            }
            cvResetImageROI(src);
        }
    }
    return smoothRegion;
}

void deConvolution::getGradient(CvMat* src, CvMat* dst, gradDirection dir)
{
    int dx, dy;
    switch(dir)
    {
        case kXdirection:
            dx = 1; dy = 0; break;
        case kYdirection:
            dx = 0; dy = 1; break;
    }
    for (int y = 0; y < dst->rows - dy; y++)
    {
        for (int x = 0; x < dst->cols - dx; x++)
        {
            cvmSet(dst, y, x, cvmGet(src, y+dy, x+dx) - cvmGet(src, y, x));
        }
    }
}

void deConvolution::startFFT(CvMat* real, CvMat* imag, CvMat* region)
{
    cvSetZero(imag);
    cvMerge(real, imag, NULL, NULL, region);
    cvDFT(region, region, CV_DXT_FORWARD, 0);
    cvSplit(region, real, imag, NULL, NULL);
}

void deConvolution::startInvFFT(CvMat* real, CvMat* imag, CvMat* region)
{
    cvMerge(real, imag, NULL, NULL, region);
    cvDFT(region, region, CV_DXT_INV_SCALE, 0);
    cvSplit(region, real, imag, NULL, NULL);
    cvPow(real, real, 2.0);
    cvPow(imag, imag, 2.0);
    cvAdd(real, imag, real, NULL);
    cvPow(real, real, 0.5);
}

void deConvolution::kernelConvertScale(CvMat* kernel, CvMat* scaledKernel)
{
    cvSetZero(scaledKernel);
    for (int y = 0; y < kernelHeight; y++)
    {
        for (int x = 0; x < kernelWidth; x++)
        {
            int h = y - (kernelHeight >> 1);
            int w = x - (kernelWidth >> 1);
            h += (h < 0)?height:0;
            w += (w < 0)?width:0;
            cvmSet(scaledKernel, h, w, cvmGet(kernel, y, x));
        }
    }
}

double deConvolution::calcPsi(double x, double m, double i, double l, int flag)
{
    double a = 6.1e-4;
    double b = 5.0;
    double k = 2.7;
    double res;

    switch(flag)
    {
        case 0:
        case 1:
            res = lambdas[0]*k*fabs(x);
            break;
        case 2:
            res = lambdas[0]*(a*pow(x, 2.0)+b);
            break;
    }
    res += lambdas[1]*m*pow(x-i, 2.0)+gamma*pow(x-l, 2.0);
    return res;
}

void deConvolution::updatingPsi(CvMat *nxt_psi, CvMat *d_blurred, CvMat *d_original, CvMat *_smoothRegion, double *_lambda, double _gamma)
{
    double a = 6.1e-4;
    double b = 5.0;
    double k = 2.7;
    double l_t = 1.8526;

    for (int y = 0; y < nxt_psi->rows; y++)
    {
        for (int x = 0; x < nxt_psi->cols; x++)
        {
            double M_i = cvmGet(_smoothRegion, y, x);
            double I_i = cvmGet(d_blurred, y, x);
            double L_i = cvmGet(d_original, y, x);

            double psi_star[3];
            psi_star[0] = (lambdas[1]*M_i*I_i + gamma*L_i + lambdas[0]*k/2.0) / (lambdas[1]*M_i + gamma);
            psi_star[1] = (lambdas[1]*M_i*I_i + gamma*L_i - lambdas[0]*k/2.0) / (lambdas[1]*M_i + gamma);
            psi_star[2] = (lambdas[1]*M_i*I_i + gamma*L_i) / (lambdas[0]*a + lambdas[1]*M_i + gamma);

            double Energy[3];
            double MinE;
            double Minzer;

            if (-l_t <= psi_star[0] && psi_star[0] <= 0)
            {
                Energy[0] = calcPsi(psi_star[0], M_i, I_i, L_i, 0);
                Minzer = psi_star[0];
            }
            else if (psi_star[0] < -l_t)
            {
                Energy[0] = calcPsi(-l_t, M_i, I_i, L_i, 0);
                Minzer = -l_t;
            }
            else if (psi_star[0] > 0)
            {
                Energy[0] = calcPsi(0, M_i, I_i, L_i, 0);
                Minzer = 0;
            }
            MinE = Energy[0];

            if (0 <= psi_star[1] && psi_star[1] <= l_t)
            {
                Energy[1] = calcPsi(psi_star[1], M_i, I_i, L_i, 1);
                if (Energy[1] < MinE)
                {
                    Minzer = psi_star[1];
                    MinE = Energy[1];
                }
            }
            else if (psi_star[1] < 0)
            {
                Energy[1] = calcPsi(0, M_i, I_i, L_i, 1);
                if (Energy[1] < MinE)
                {
                    Minzer = 0;
                    MinE = Energy[1];
                }
            }
            else if (psi_star[1] > l_t)
            {
                Energy[1] = calcPsi(l_t, M_i, I_i, L_i, 1);
                if (Energy[1] < MinE)
                {
                    Minzer = l_t;
                    MinE = Energy[1];
                }
            }

            if (l_t < psi_star[2] && psi_star[2] < -l_t)
            {
                Energy[2] = calcPsi(psi_star[2], M_i, I_i, L_i, 2);
                if (Energy[2] < MinE)
                {
                    Minzer = psi_star[2];
                    MinE = Energy[2];
                }
            }
            else
            {
                double tmp1 = calcPsi(l_t, M_i, I_i, L_i, 2);
                double tmp2 = calcPsi(-l_t, M_i, I_i, L_i, 2);
                if (tmp1 < MinE) Minzer = l_t;
                else if (tmp2 < MinE) Minzer = -l_t;
            }

            cvmSet(nxt_psi, y, x, Minzer);
        }
    }
}

void deConvolution::updatingOriginal(CvMat* _psi_x, CvMat* _psi_y, CvMat* _kernel, CvMat* nxt_L)
{
    cvConvert(_psi_x, F_psi_x[0]);
    startFFT(F_psi_x[0], F_psi_x[1], FFTRegion);
    cvConvert(_psi_y, F_psi_y[0]);
    startFFT(F_psi_y[0], F_psi_y[1], FFTRegion);

    CvMat *FL_star[2];
    FL_star[0] = cvCreateMat(height, width, CV_64FC1);
    FL_star[1] = cvCreateMat(height, width, CV_64FC1);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double r, i;
            r = cvmGet(F_blurred[0], y, x);
            i = cvmGet(F_blurred[1], y, x);
            complex<double> FI(r, i);
            r = cvmGet(F_kernel[0], y, x);
            i = cvmGet(F_kernel[1], y, x);
            complex<double> Ff(r, i);
            r = cvmGet(F_grad_x[0], y, x);
            i = cvmGet(F_grad_x[1], y, x);
            complex<double> Fdx(r, i);
            r = cvmGet(F_grad_y[0], y, x);
            i = cvmGet(F_grad_y[1], y, x);
            complex<double> Fdy(r, i);
            r = cvmGet(F_psi_x[0], y, x);
            i = cvmGet(F_psi_x[1], y, x);
            complex<double> Fpsi_x(r, i);
            r = cvmGet(F_psi_y[0], y, x);
            i = cvmGet(F_psi_y[1], y, x);
            complex<double> Fpsi_y(r, i);
            double Fdelta = cvmGet(F_delta, y, x);
            complex<double> L_star = (conj(Ff)*FI*Fdelta + gamma*conj(Fdx)*Fpsi_x + gamma*conj(Fdy)*Fpsi_y) / (norm(Ff)*Fdelta + gamma*norm(Fdx) + gamma*norm(Fdy));

            cvmSet(FL_star[0], y, x, real(L_star));
            cvmSet(FL_star[1], y, x, imag(L_star));
        }
    }
    startInvFFT(FL_star[0], FL_star[1], FFTRegion);
    cvConvert(FL_star[0], nxt_L);

    cvReleaseMat(&FL_star[0]);  cvReleaseMat(&FL_star[1]);
}

void deConvolution::deConvolutionChannel(double tol, int maxSteps)
{
    cvConvert(blurred, original);
    int iterCount = 0;
    do
    {
        double delta_psi = DOUBLE_MAX, delta_L = DOUBLE_MAX;
        iterCount++;
        kernelConvertScale(kernel, F_kernel[0]);
        startFFT(F_kernel[0], F_kernel[1], FFTRegion);
        do
        {
            cvConvert(original, prev_original);
            cvConvert(psi_x, prev_psi_x);
            cvConvert(psi_y, prev_psi_y);

            getGradient(original, L_grad_x, kXdirection);
            getGradient(original, L_grad_y, kYdirection);
            updatingPsi(psi_x, I_grad_x, L_grad_x, SmoothRegion, lambdas, gamma);
            //cout << "delta_psi_x = " << cvNorm(psi_x, prev_psi_x, CV_L2, NULL) << endl;
            updatingPsi(psi_y, I_grad_y, L_grad_y, SmoothRegion, lambdas, gamma);
            //cout << "delta_psi_x = " << cvNorm(psi_y, prev_psi_y, CV_L2, NULL) << endl;

            updatingOriginal(psi_x, psi_y, kernel, original);
            delta_L = cvNorm(original, prev_original, CV_L2, NULL);
            cout << "delta_L = " << delta_L << endl;
            double x = cvNorm(psi_x, prev_psi_x, CV_L2, NULL);
            double y = cvNorm(psi_y, prev_psi_y, CV_L2, NULL);
            delta_psi = sqrt(x*x + y*y);
            cout << "delta_psi = " << delta_psi << endl;
            gamma *= 2.0;
        }while(delta_psi >= tol || delta_L >= tol);

        cvConvert(kernel, prev_kernel);
        compressedSense *cs = new compressedSense(kernelHeight*kernelWidth, height*width);
        cs->Mat2Vector(original, blurred, imgSize, kernelSize);
        cs->solveOptm();
        cs->Vector2Mat(kernel);

        double delta_kernel = cvNorm(kernel, prev_kernel, CV_L2, NULL);
        if (delta_kernel < tol)
        {
            cout << "iteration done!" << endl;
            break;
        }
        else
        {
            lambdas[0] /= kappa_1;
            lambdas[1] /= kappa_2;
        }
    }while (iterCount <= maxSteps);
}

void deConvolution::getResult(IplImage* res)
{
    cvConvertScale(original, res, 2048.0, 0);
}
