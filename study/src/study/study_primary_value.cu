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

#ifdef CUDA_COMPILE
    cudaMalloc((void**)&d_listCalculatedBestResultValue, sizeof(double)*allocatedMax);
    cudaMalloc((void**)&d_listCalculatedDotProductBestRateValues, sizeof(double)*allocatedMax*MAX_MATRIX_COLUMS);
    cudaMalloc((void**)&d_globalBestResultMax, sizeof(double));
    cudaMalloc((void**)&d_listResultColumnIndex, sizeof(int)*listResultColumnIndexCount);
    cudaMemcpy(d_listResultColumnIndex, listResultColumnIndex, sizeof(int)*listResultColumnIndexCount, cudaMemcpyHostToDevice);
#else
    d_listCalculatedBestResultValue = new double[allocatedMax];
    d_listCalculatedDotProductBestRateValues = new double[allocatedMax*MAX_MATRIX_COLUMS];
    d_globalBestResultMax = new double;
    d_listResultColumnIndex = new int[listResultColumnIndexCount];
    memcpy(d_listResultColumnIndex, listResultColumnIndex, sizeof(int)*listResultColumnIndexCount);
#endif
    for (i2=0;i2<allocatedMax;i2++) {
        listCalculatedBestResultValue[i2] = DEFAULT_BEST_VALUE;
    }

    double tmp = bestResultStorage->getCurrentBestResult();
#ifdef CUDA_COMPILE
    cudaMemcpy(d_globalBestResultMax, &tmp, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_listCalculatedBestResultValue, listCalculatedBestResultValue, sizeof(double) * allocatedMax, cudaMemcpyHostToDevice);
    studyBestResultPrimaryValue<<<1, 1>>>(d_listTableData,
                    d_listCalculatedBestResultValue,
                    d_listCalculatedDotProductBestRateValues,
                    d_globalBestResultMax,
                    d_listResultColumnIndex,
                    listResultColumnIndexCount);
    cudaDeviceSynchronize();
    cudaMemcpy(listCalculatedBestResultValue, d_listCalculatedBestResultValue, sizeof(double) * allocatedMax, cudaMemcpyDeviceToHost);
    cudaMemcpy(listCalculatedDotProductBestRateValues, d_listCalculatedDotProductBestRateValues,  sizeof(double)*allocatedMax*MAX_MATRIX_COLUMS, cudaMemcpyDeviceToHost);
#else
    memcpy(d_globalBestResultMax, &tmp, sizeof(double));
    memcpy(d_listCalculatedBestResultValue, listCalculatedBestResultValue, sizeof(double) * allocatedMax);
    studyBestResultPrimaryValue(d_listTableData,
                    d_listCalculatedBestResultValue,
                    d_listCalculatedDotProductBestRateValues,
                    d_globalBestResultMax,
                    d_listResultColumnIndex,
                    listResultColumnIndexCount);
    memcpy(listCalculatedBestResultValue, d_listCalculatedBestResultValue, sizeof(double) * allocatedMax);
    memcpy(listCalculatedDotProductBestRateValues, d_listCalculatedDotProductBestRateValues,  sizeof(double)*allocatedMax*MAX_MATRIX_COLUMS);
#endif
    bestResultStorage->setBestResultPrimaryOnly(
                        listCalculatedBestResultValue[0], listResultColumnIndex[0],
                        listCalculatedDotProductBestRateValues);
    delete[] listCalculatedBestResultValue;
    delete[] listCalculatedDotProductBestRateValues;
#ifdef CUDA_COMPILE
    cudaFree(d_listCalculatedBestResultValue);
    cudaFree(d_listCalculatedDotProductBestRateValues);
    cudaFree(d_globalBestResultMax);
    cudaFree(d_listResultColumnIndex);
#else
    delete[] d_listCalculatedBestResultValue;
    delete[] d_listCalculatedDotProductBestRateValues;
    delete d_globalBestResultMax;
    delete[] d_listResultColumnIndex;
#endif
}

}
