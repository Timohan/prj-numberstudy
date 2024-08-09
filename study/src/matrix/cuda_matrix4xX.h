/**
 * @file cuda_matrix4xX.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief matrix 4x4 line calculations
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef CUDA_MATRIX_4XX_H_
#define CUDA_MATRIX_4XX_H_

#include "../define_values.h"
#include "cuda_matrix4x4.h"

namespace CudaMatrix4xX
{
    typedef struct {
        CudaMatrix4x4::Matrix4x1 *m_listMatrixLines;
        double *m_listValue;
        unsigned int m_listMatrixLinesCount;
    } Matrix4xX;

    typedef struct {
        Matrix4xX *m_listMatrix4xX;
        unsigned int m_listMatrix4xXCount;
    } ListMatrix4xX;

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void calculateDotProducts(const Matrix4xX *listMatrix, CudaMatrix4x4::Matrix4x1 *listResultOut, unsigned int *lineResultOutLineIndex, unsigned int *listResultCountOut);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
bool calculateDotProductsNextLine(const Matrix4xX *listMatrix, CudaMatrix4x4::Matrix4x1 *listResultOut,
                                  unsigned int *lineIndex);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void init(CudaMatrix4xX::Matrix4xX *matrix, unsigned int maxMatrixLineCount);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void clear(CudaMatrix4xX::Matrix4xX *matrix);


};

#endif // CUDA_MATRIX_4XX_H_