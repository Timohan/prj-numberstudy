/**
 * @file cuda_matrix4xX.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief matrix 4x4 line calculations
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "cuda_matrix4xX.h"
#include <stdio.h>

namespace CudaMatrix4xX
{

/*! @brief
 * calculates 4x4 matrix inversion & dot product for each line
 * @param listMatrix [in] values (lines) to calculate 4x4 matrix inversion & dot product
 * @param listResultOut [out] list of inversion & dot product from each 4x4 listMatrix line
 * 1st line contains results from lines 0-3 (assuming this line's inversion was possible, if not, then lines 1-4)
 * next line contains results from lines 1-4 (assuming this line's inversion was possible, if not, then next lines 2-5)
 * @param lineResultOutLineIndex [out] listResultOut's line indexes of listMatrix
 * @param listResultCountOut [out] number of line results calculated to listResultOut
 */
#ifdef CUDA_COMPILE
__host__ __device__
#endif
void calculateDotProducts(const Matrix4xX *listMatrix, CudaMatrix4x4::Matrix4x1 *listResultOut, unsigned int *lineResultOutLineIndex, unsigned int *listResultCountOut)
{
    unsigned int i, x, y;
    CudaMatrix4x4::Matrix4x4 m4x4;
    CudaMatrix4x4::Matrix4x4 m4x4Inverted;
    CudaMatrix4x4::Matrix1x4 m1x4;
    CudaMatrix4x4::Matrix4x1 m1x4Result;
    unsigned int index = 0;

    for (i=MAX_MATRIX_COLUMS-1;i<listMatrix->m_listMatrixLinesCount;i++) {
        for (y=0;y<MAX_MATRIX_COLUMS;y++) {
            for (x=0;x<MAX_MATRIX_COLUMS;x++) {
                CudaMatrix4x4::setMatrixValue(&m4x4, x, y, listMatrix->m_listMatrixLines[i-3+y].m[x]);
            }
            m1x4.m[y] = listMatrix->m_listValue[i-3+y];
        }

        if (!CudaMatrix4x4::inversion(&m4x4, &m4x4Inverted)) {
            continue;
        }

        CudaMatrix4x4::dotProduct(&m4x4Inverted, &m1x4, &m1x4Result);
        for (x=0;x<MAX_MATRIX_COLUMS;x++) {
            listResultOut[index].m[x] = m1x4Result.m[x];
        }
        lineResultOutLineIndex[index] = i;
        index++;
    }
    *listResultCountOut = index;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief calculates 4x4 dot product for next valid line from listMatrix
 * into listResultOut
 * 
 * @param listMatrix [in] list of 4x4 matrixes
 * @param listResultOut [out] dot product results
 * @param lineIndex [in/out] from line to start calculation, next line out
 * @return true if calculation did success.
 * @return false if there was no possible to calculate dot product
 */
bool calculateDotProductsNextLine(const Matrix4xX *listMatrix, CudaMatrix4x4::Matrix4x1 *listResultOut,
                                  unsigned int *lineIndex)
{
    unsigned int i, x, y;
    CudaMatrix4x4::Matrix4x4 m4x4;
    CudaMatrix4x4::Matrix4x4 m4x4Inverted;
    CudaMatrix4x4::Matrix1x4 m1x4;
    unsigned int index = *lineIndex;

    for (i=index;i<listMatrix->m_listMatrixLinesCount;i++) {
        for (y=0;y<MAX_MATRIX_COLUMS;y++) {
            for (x=0;x<MAX_MATRIX_COLUMS;x++) {
                CudaMatrix4x4::setMatrixValue(&m4x4, x, y, listMatrix->m_listMatrixLines[i-3+y].m[x]);
            }
            m1x4.m[y] = listMatrix->m_listValue[i-3+y];
        }

        if (!CudaMatrix4x4::inversion(&m4x4, &m4x4Inverted)) {
            continue;
        }

        CudaMatrix4x4::dotProduct(&m4x4Inverted, &m1x4, listResultOut);
        *lineIndex = i+1;
        return true;
    }
    return false;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief init matrix lines
 *
 * @param matrix [in/out] allocate pointers here
 * @param maxMatrixLineCount [in] allocated count
 */
void init(CudaMatrix4xX::Matrix4xX *matrix, unsigned int maxMatrixLineCount)
{
    matrix->m_listMatrixLines = new CudaMatrix4x4::Matrix4x1[maxMatrixLineCount];
    matrix->m_listValue = new double[maxMatrixLineCount];
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief clears pointers from matrixLines
 *
 * @param matrix [in/out] deletes pointers from this matrix lines
 */
void clear(CudaMatrix4xX::Matrix4xX *matrix)
{
    delete[] matrix->m_listMatrixLines;
    delete[] matrix->m_listValue;
}

};
