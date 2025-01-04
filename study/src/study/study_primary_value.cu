/**
 * @file study_primary_value.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief calculate primary value's start values
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "study_primary_value.h"
#include "../study_cuda/study_best_result_primary_value.h"
#include "../loader/best_result_storage.h"
#include <stdio.h>
#include "../macros.h"

#ifndef CUDA_COMPILE
#include <cstddef>
#include <cstring>
#endif

namespace StudyPrimaryValue
{

#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief calculate primary value's start values
 * 
 * @param d_listTableData pointer to list table data in gpu
 * @param listResultColumnIndex list of columns to find new counter part
 * @param listResultColumnIndexCount list of columns count
 * @param bestResultStorage current best result
 */
void study(ListTableData *d_listTableData, const int *listResultColumnIndex,
                    const int listResultColumnIndexCount, BestResultStorage *bestResultStorage)
{
    uint64_t partIndexMax = 1;
    uint64_t i2, allocatedMax = partIndexMax;
    double *listCalculatedBestResultValue = new double[allocatedMax];
    double *listCalculatedDotProductBestRateValues = new double[allocatedMax*MAX_MATRIX_COLUMS];
    double *d_listCalculatedBestResultValue;
    double *d_listCalculatedDotProductBestRateValues;
    double *d_globalBestResultMax;
    int *d_listResultColumnIndex;

    PRJ_ALLOC(d_listCalculatedBestResultValue, double, allocatedMax);
    PRJ_ALLOC(d_listCalculatedDotProductBestRateValues, double, allocatedMax*MAX_MATRIX_COLUMS);
    PRJ_ALLOC(d_globalBestResultMax, double, 1);
    PRJ_ALLOC(d_listResultColumnIndex, int, listResultColumnIndexCount);
    PRJ_MEMCPY(d_listResultColumnIndex, listResultColumnIndex, sizeof(int)*listResultColumnIndexCount, cudaMemcpyHostToDevice);
    for (i2=0;i2<allocatedMax;i2++) {
        listCalculatedBestResultValue[i2] = DEFAULT_BEST_VALUE;
    }

    double tmp = bestResultStorage->getCurrentBestResult();
    PRJ_MEMCPY(d_globalBestResultMax, &tmp, sizeof(double), cudaMemcpyHostToDevice);
    PRJ_MEMCPY(d_listCalculatedBestResultValue, listCalculatedBestResultValue, sizeof(double)*allocatedMax, cudaMemcpyHostToDevice);
    PRJ_FUNC_CALL_SINGLE(studyBestResultPrimaryValue, d_listTableData,
                    d_listCalculatedBestResultValue,
                    d_listCalculatedDotProductBestRateValues,
                    d_globalBestResultMax,
                    d_listResultColumnIndex,
                    listResultColumnIndexCount);
    PRJ_CUDA_WAIT();
    PRJ_MEMCPY(listCalculatedBestResultValue, d_listCalculatedBestResultValue, sizeof(double)*allocatedMax, cudaMemcpyDeviceToHost);
    PRJ_MEMCPY(listCalculatedDotProductBestRateValues, d_listCalculatedDotProductBestRateValues, sizeof(double)*allocatedMax*MAX_MATRIX_COLUMS, cudaMemcpyDeviceToHost);
    bestResultStorage->setBestResultPrimaryOnly(
                        listCalculatedBestResultValue[0], listResultColumnIndex[0],
                        listCalculatedDotProductBestRateValues);
    delete[] listCalculatedBestResultValue;
    delete[] listCalculatedDotProductBestRateValues;
    PRJ_FREE(d_listCalculatedBestResultValue)
    PRJ_FREE(d_listCalculatedDotProductBestRateValues)
    PRJ_FREE(d_globalBestResultMax)
    PRJ_FREE(d_listResultColumnIndex)
}

}
