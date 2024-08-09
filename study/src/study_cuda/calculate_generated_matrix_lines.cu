/**
 * @file calculate_generated_matrix_lines.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief calculate generated matrix lines
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "calculate_generated_matrix_lines.h"
#include <stdio.h>

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief calculates dot result with matrix line
 * 
 * @param line calculated 4x1 dot result
 * @param matrixLines [in] matrix lines that are used for calculation
 * @param baseMatrixLineIndex index of line from matrixLines
 * @return double calculated result
 */
double calculateMatrix4xXLine(const CudaMatrix4x4::Matrix4x1 *line,
                              const CudaMatrix4xX::Matrix4xX *matrixLines,
                              const unsigned int baseMatrixLineIndex)
{
    double ret = 0;
    for (unsigned int i=0;i<MAX_MATRIX_COLUMS;i++) {
        ret += matrixLines->m_listMatrixLines[baseMatrixLineIndex].m[i]*line->m[i];
    }
    return ret;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/** @brief 
 * calculate best possible best result
 * @param localBestResult [in/out] if new best result if found, it is filled here
 * with details
 * @param globalBestResultMax [in/out] global best result value,
 * if new best result value if found, it is filled here
 * @param matrixLines [in] matrix lines that are used for calculation
 * @param matrixLineIndexForDotProducts
 * @return true if new best result is found so calling function can set bestResult param's values
 * that are not changing in this function
 */
bool calculateGeneratedMatrixLine(double *localBestResult, double *globalBestResultMax,
                    const CudaMatrix4xX::ListMatrix4xX *listMatrixLines,
                    double calculatedDotProductValues[MAX_MATRIX_COLUMS],
                    const unsigned int matrixLineIndexForDotProducts)
{
    unsigned int i;
    unsigned int resultIndex;
    double calculatedResult;
    bool ret = false;
    double result;
    CudaMatrix4x4::Matrix4x1 *listResultOut;
    unsigned int *lineResultOutLineIndex;
    unsigned int listResultCount;
    bool failed = false;
    unsigned int matrixLineIndex;
    listResultOut = new CudaMatrix4x4::Matrix4x1[listMatrixLines->m_listMatrix4xX[matrixLineIndexForDotProducts].m_listMatrixLinesCount+1];
    lineResultOutLineIndex = new unsigned int[listMatrixLines->m_listMatrix4xX[matrixLineIndexForDotProducts].m_listMatrixLinesCount+1];

    calculateDotProducts(&listMatrixLines->m_listMatrix4xX[matrixLineIndexForDotProducts], listResultOut, lineResultOutLineIndex, &listResultCount);

    for (resultIndex=0;resultIndex<listResultCount;resultIndex++) {
        calculatedResult = 0;
        failed = false;
        for (matrixLineIndex=0;matrixLineIndex<listMatrixLines->m_listMatrix4xXCount && !failed;matrixLineIndex++) {
            for (i=0;i<listMatrixLines->m_listMatrix4xX[matrixLineIndex].m_listMatrixLinesCount && !failed;i++) {
#if MAX_MATRIX_COLUMS==4
                if (matrixLineIndex == matrixLineIndexForDotProducts
                    && (lineResultOutLineIndex[resultIndex] == i+MAX_MATRIX_COLUMS-1
                        || lineResultOutLineIndex[resultIndex] == i+MAX_MATRIX_COLUMS-2
                        || lineResultOutLineIndex[resultIndex] == i+MAX_MATRIX_COLUMS-3
                        || lineResultOutLineIndex[resultIndex] == i+MAX_MATRIX_COLUMS-4)) {
                    continue;
                }
#else
                printf("Matrix columns incorrectly handled %s %d.\n", __FILE__, __LINE__);
#endif
                result = calculateMatrix4xXLine(&listResultOut[resultIndex],
                              &listMatrixLines->m_listMatrix4xX[matrixLineIndex],
                              i);
                result = (result-listMatrixLines->m_listMatrix4xX[matrixLineIndex].m_listValue[i]);
                if (result < 0) {
                    result = result*-1;
                }
                if (calculatedResult < result) {
                    calculatedResult = result;
                }
               if (calculatedResult >= *localBestResult
                    || calculatedResult >= *globalBestResultMax) {
                    failed = true;
                    break;
                }
            }
        }
        if (calculatedResult < *localBestResult
            && calculatedResult < *globalBestResultMax) {
            *globalBestResultMax = calculatedResult;
            *localBestResult = calculatedResult;
            for (i=0;i<MAX_MATRIX_COLUMS;i++) {
                calculatedDotProductValues[i] = listResultOut[resultIndex].m[i];
            }
            ret = true;
        }
    }

    delete[] listResultOut;
    delete[] lineResultOutLineIndex;
    return ret;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/** @brief
 * calculate best possible best result
 * @param localBestResult [in/out] if new best result if found, it is filled here
 * with details
 * @param globalBestResultMax [in/out] global best result value,
 * if new best result value if found, it is filled here
 * @param matrixLines [in] matrix lines that are used for calculation
 * @return true if new best result is found so calling function can set bestResult param's values
 * that are not changing in this function
 */
bool calculateGeneratedMatrixLines(double *localBestResult, double *globalBestResultMax,
                    const CudaMatrix4xX::ListMatrix4xX *listMatrixLines,
                    double calculatedDotProductValues[MAX_MATRIX_COLUMS])
{
    bool ret = false;
    unsigned int i;
    for (i=0;i<listMatrixLines->m_listMatrix4xXCount;i++) {
        if (calculateGeneratedMatrixLine(localBestResult, globalBestResultMax,
                    listMatrixLines,
                    calculatedDotProductValues,
                    i)) {
            ret = true;
        }
    }

    return ret;
}
