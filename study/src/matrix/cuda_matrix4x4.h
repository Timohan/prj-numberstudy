/**
 * @file cuda_matrix4x4.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief matrix 4x4 math
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef CUDA_MATRIX_4X4_H_
#define CUDA_MATRIX_4X4_H_

namespace CudaMatrix4x4
{
    struct Matrix4x4
    {
        double m[16];
    };

    struct Matrix1x4
    {
        double m[4];
    };

    struct Matrix4x1
    {
        double m[4];
    };

#ifdef CUDA_COMPILE
__host__ __device__
#endif
bool inversion(const Matrix4x4 *inputMatrix, Matrix4x4 *dst);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
double getMatrixValue(const Matrix4x4 *m, unsigned int x, unsigned int y);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void setMatrixValue(Matrix4x4 *m, unsigned int x, unsigned int y, double value);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void setMatrixValue(Matrix1x4 *m, unsigned int y, double value);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
double getMatrixValue(const Matrix1x4 *m, unsigned int y);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void dotProduct(const Matrix4x4 *m4x4, const Matrix1x4 *m1x4, Matrix4x1 *dst);
};

#endif // CUDA_MATRIX_4X4_H_