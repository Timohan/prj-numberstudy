/**
 * @file study_best_result.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief general calculate/study start dot products and result value
 *
 * @copyright Copyright (c) 2024
 * 
 */
#include "study_best_result.h"
#include "generate_matrix_lines.h"
#include "calculate_generated_matrix_lines.h"
#include <stdio.h>

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief study/calculate best result
 * 
 * @param globalBestResultMax [in/out] global best result
 * @param localBestResult [in/out] local best result
 * @param calculatedDotProductValues [out] calculated dot product values
 * @param list table data list
 * @param bestCounterPartMultipliersPrimary current best multipliers for primary table
 * @param bestCounterPartMultipliersCounter current best multipliers for counter table
 * @param previousIndex previous indexes for counter tables
 * @param listResultColumnIndex list of column indexes to calculate best result value
 * @param listResultColumnIndexCount list of column indexes count
 * @param listCounterPartIndex list of counter part indexes
 * @param listCounterPartMathType list of counter part math types
 * @return true 
 * @return false 
 */
bool studyBestResult(double *globalBestResultMax,
                    double *localBestResult,
                    double calculatedDotProductValues[MAX_MATRIX_COLUMS],
                    const ListTableData *list,
                    const int bestCounterPartMultipliersPrimary[MAX_MATRIX_COLUMS],
                    const int bestCounterPartMultipliersCounter[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT],
                    const int *previousIndex,
                    const int *listResultColumnIndex,
                    const int listResultColumnIndexCount,
                    const int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                    const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT])
{
    bool ret = true;
    unsigned int rowCount = getMaxRowCountFromListTableData(list);
    CudaMatrix4xX::ListMatrix4xX listMatrixLines;
    listMatrixLines.m_listMatrix4xXCount = listResultColumnIndexCount;
    listMatrixLines.m_listMatrix4xX = new CudaMatrix4xX::Matrix4xX[listResultColumnIndexCount];
    for (int i=0;i<listResultColumnIndexCount;i++) {
        CudaMatrix4xX::init(&listMatrixLines.m_listMatrix4xX[i], rowCount);
        listMatrixLines.m_listMatrix4xX[i].m_listMatrixLinesCount = 0;
        if (!generateMatrixLines(list,
                             bestCounterPartMultipliersPrimary,
                             bestCounterPartMultipliersCounter,
                             previousIndex,
                             listResultColumnIndex[i],
                             listCounterPartIndex,
                             &listMatrixLines.m_listMatrix4xX[i],
                             listCounterPartMathType)) {
            ret = false;
            break;
        }
    }

    if (ret) {
        ret = calculateGeneratedMatrixLines(localBestResult, globalBestResultMax, &listMatrixLines, calculatedDotProductValues);
    }
    for (int i=0;i<listResultColumnIndexCount;i++) {
        CudaMatrix4xX::clear(&listMatrixLines.m_listMatrix4xX[i]);
    }
    delete[]listMatrixLines.m_listMatrix4xX;
    return ret;
}
