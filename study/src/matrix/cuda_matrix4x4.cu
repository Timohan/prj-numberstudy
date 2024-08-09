/**
 * @file cuda_matrix4x4.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief matrix 4x4 math
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "cuda_matrix4x4.h"

namespace CudaMatrix4x4
{
#ifdef CUDA_COMPILE
__host__ __device__
#endif
double getMatrixValue(const Matrix4x4 *m, unsigned int x, unsigned int y)
{
    return m->m[y*4+x];
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void setMatrixValue(Matrix4x4 *m, unsigned int x, unsigned int y, double value)
{
    m->m[y*4+x] = value;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
double getMatrixValue(const Matrix1x4 *m, unsigned int y)
{
    return m->m[y];
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void setMatrixValue(Matrix1x4 *m, unsigned int y, double value)
{
    m->m[y] = value;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
bool inversion(const Matrix4x4 *inputMatrix, Matrix4x4 *dst)
{
    unsigned int i;
    double tmp[12];
    double src[16];
    double det;

    for (i=0;i<4;i++) {
        src[i] = inputMatrix->m[i*4];
        src[i+4] = inputMatrix->m[i*4+1];
        src[i+8] = inputMatrix->m[i*4+2];
        src[i+12] = inputMatrix->m[i*4+3];
    }

    tmp[0] = src[10] * src[15];
    tmp[1] = src[11] * src[14];
    tmp[2] = src[9] * src[15];
    tmp[3] = src[11] * src[13];
    tmp[4] = src[9] * src[14];
    tmp[5] = src[10] * src[13];
    tmp[6] = src[8] * src[15];
    tmp[7] = src[11] * src[12];
    tmp[8] = src[8] * src[14];
    tmp[9] = src[10] * src[12];
    tmp[10] = src[8] * src[13];
    tmp[11] = src[9] * src[12];

    dst->m[0] = tmp[0]*src[5] + tmp[3]*src[6] + tmp[4]*src[7];
    dst->m[0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[5]*src[7];
    dst->m[1] = tmp[1]*src[4] + tmp[6]*src[6] + tmp[9]*src[7];
    dst->m[1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[8]*src[7];
    dst->m[2] = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
    dst->m[2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
    dst->m[3] = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
    dst->m[3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
    dst->m[4] = tmp[1]*src[1] + tmp[2]*src[2] + tmp[5]*src[3];
    dst->m[4] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[4]*src[3];
    dst->m[5] = tmp[0]*src[0] + tmp[7]*src[2] + tmp[8]*src[3];
    dst->m[5] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[9]*src[3];
    dst->m[6] = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
    dst->m[6] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
    dst->m[7] = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
    dst->m[7] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];

    tmp[0] = src[2]*src[7];
    tmp[1] = src[3]*src[6];
    tmp[2] = src[1]*src[7];
    tmp[3] = src[3]*src[5];
    tmp[4] = src[1]*src[6];
    tmp[5] = src[2]*src[5];
    tmp[6] = src[0]*src[7];
    tmp[7] = src[3]*src[4];
    tmp[8] = src[0]*src[6];
    tmp[9] = src[2]*src[4];
    tmp[10] = src[0]*src[5];
    tmp[11] = src[1]*src[4];

    dst->m[8] = tmp[0]*src[13] + tmp[3]*src[14] + tmp[4]*src[15];
    dst->m[8] -= tmp[1]*src[13] + tmp[2]*src[14] + tmp[5]*src[15];
    dst->m[9] = tmp[1]*src[12] + tmp[6]*src[14] + tmp[9]*src[15];
    dst->m[9] -= tmp[0]*src[12] + tmp[7]*src[14] + tmp[8]*src[15];
    dst->m[10] = tmp[2]*src[12] + tmp[7]*src[13] + tmp[10]*src[15];
    dst->m[10]-= tmp[3]*src[12] + tmp[6]*src[13] + tmp[11]*src[15];
    dst->m[11] = tmp[5]*src[12] + tmp[8]*src[13] + tmp[11]*src[14];
    dst->m[11]-= tmp[4]*src[12] + tmp[9]*src[13] + tmp[10]*src[14];
    dst->m[12] = tmp[2]*src[10] + tmp[5]*src[11] + tmp[1]*src[9];
    dst->m[12]-= tmp[4]*src[11] + tmp[0]*src[9] + tmp[3]*src[10];
    dst->m[13] = tmp[8]*src[11] + tmp[0]*src[8] + tmp[7]*src[10];
    dst->m[13]-= tmp[6]*src[10] + tmp[9]*src[11] + tmp[1]*src[8];
    dst->m[14] = tmp[6]*src[9] + tmp[11]*src[11] + tmp[3]*src[8];
    dst->m[14]-= tmp[10]*src[11] + tmp[2]*src[8] + tmp[7]*src[9];
    dst->m[15] = tmp[10]*src[10] + tmp[4]*src[8] + tmp[9]*src[9];
    dst->m[15]-= tmp[8]*src[9] + tmp[11]*src[10] + tmp[5]*src[8];

    det=src[0]*dst->m[0]+src[1]*dst->m[1]+src[2]*dst->m[2]+src[3]*dst->m[3];
    if (det == 0) {
        return false;
    }

    det = 1/det;
    for (i=0;i<16;i++) {
        dst->m[i] *= det;
    }
    return true;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void dotProduct(const Matrix4x4 *m4x4, const Matrix1x4 *m1x4, Matrix4x1 *dst)
{
    dst->m[0] = m4x4->m[0]*m1x4->m[0] + m4x4->m[1]*m1x4->m[1] + m4x4->m[2]*m1x4->m[2] + m4x4->m[3]*m1x4->m[3];
    dst->m[1] = m4x4->m[4]*m1x4->m[0] + m4x4->m[5]*m1x4->m[1] + m4x4->m[6]*m1x4->m[2] + m4x4->m[7]*m1x4->m[3];
    dst->m[2] = m4x4->m[8]*m1x4->m[0] + m4x4->m[9]*m1x4->m[1] + m4x4->m[10]*m1x4->m[2] + m4x4->m[11]*m1x4->m[3];
    dst->m[3] = m4x4->m[12]*m1x4->m[0] + m4x4->m[13]*m1x4->m[1] + m4x4->m[14]*m1x4->m[2] + m4x4->m[15]*m1x4->m[3];
}
};
