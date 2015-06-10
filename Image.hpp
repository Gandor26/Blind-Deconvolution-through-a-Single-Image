#ifndef IMAGE_H
#define IMAGE_H
#include <core/core_c.h>
#include <core/types_c.h>

template <typename T>
class Image
{
    public:
        Image(IplImage* image)
        {
            pImage = image;
        }
        virtual ~Image()
        {
            pImage = NULL;
        }

        void operator = (IplImage* image)
        {
            pImage = image;
        }
        T* operator [] (int row)
        {
            return ((T*)(pImage->imageData + pImage->widthStep*row));
        }
    private:
        IplImage* pImage;
};

typedef Image<uchar> BWImage;
#endif // IMAGE_H
