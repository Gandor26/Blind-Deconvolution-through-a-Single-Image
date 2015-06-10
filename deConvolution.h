#ifndef DECONVOLUTION_H
#define DECONVOLUTION_H

#include <cv.h>
#include <highgui.h>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <cstddef>
#include "Image.hpp"

#ifndef DOUBLE_MAX
#define DOUBLE_MAX 2147483647.0
#endif // DOUBLE_MAX
#define weightInit 50.0
#define lambda_1 0.05
#define lambda_2 20.0
#define kappa_1 1.2         //accentuate of global prior
#define kappa_2 1.5         //accentuate of local prior
#define gammaInit 2.0

typedef enum _gradDirection
{
    kXdirection,
    kYdirection
}gradDirection;

typedef enum _colorChannel
{
    Blue,
    Green,
    Red
}colorChannel;

class deConvolution
{
    public:
        //deConvolution(){}
        deConvolution(IplImage *blurredImg, CvSize kerSize);
        virtual ~deConvolution();
        void deConvolutionChannel(double tol = 1e-5, int maxSteps = 10);
        void getResult(IplImage* res);
        IplImage* findSmooth(int WindowSize, double threshold);
    protected:
    public:
        void getGradient(CvMat* src, CvMat* dst, gradDirection dir);
        void startFFT(CvMat* real, CvMat* imag, CvMat* region);
        void startInvFFT(CvMat* real, CvMat* imag, CvMat* region);
        void kernelConvertScale(CvMat* kernel, CvMat* scaledKernel);
        double calcPsi(double x, double m, double i, double l, int flag);
        void updatingPsi(CvMat *nxt_psi, CvMat *_blurred, CvMat *_original, CvMat *_smoothRegion, double *_lambda, double _gamma);
        void updatingOriginal(CvMat* _psi_x, CvMat* _psi_y, CvMat* _kernel, CvMat* nxt_L);

        int width, height;
        CvSize imgSize;
        int kernelWidth, kernelHeight;
        CvSize kernelSize;

        double lambdas[2];
        double gamma;

        IplImage *src;
        CvMat *blurred, *original, *prev_original;
        CvMat *kernel, *prev_kernel;

        CvMat *psi_x, *psi_y, *prev_psi_x, *prev_psi_y;
        CvMat *I_grad_x, *I_grad_y;
        CvMat *L_grad_x, *L_grad_y;
        CvMat *SmoothRegion, *FFTRegion;
        CvMat *F_delta;     //gradSum

        /*      Fourier Transformation   */
        CvMat *F_blurred[2];
        CvMat *F_grad_x[2], *F_grad_y[2];
        CvMat *F_psi_x[2], *F_psi_y[2];
        CvMat *F_kernel[2];
};

#endif // DECONVOLUTION_H
