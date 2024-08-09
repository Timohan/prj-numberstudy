/**
 * @file study_finetune_primary_value.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief finetune primary value search
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "study_finetune_primary_value.h"
#ifndef CUDA_COMPILE
#include <cstdlib>
#include <cstring>
#else
#include <cuda.h>
#endif
#include "../data/list_table_data.h"
#include "../study_cuda/study_best_result_finetune_primary_value.h"
#include "../loader/best_result_storage.h"
#include <stdio.h>

namespace StudyFinetunePrimaryValue
{
#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief finetune primary value
 * 
 * @param d_listTableData pointer to list table data in gpu
 * @param listResultColumnIndex list of columns to find new counter part
 * @param listResultColumnIndexCount list of columns count
 * @param bestResultStorage current best result
 */
void study(ListTableData *d_listTableData,
                                      const int *listResultColumnIndex,
                                      const int listResultColumnIndexCount,
                                      BestResultStorage *bestResultStorage)
{
    uint64_t partIndexMax = getPartIndexMax();
    uint64_t i, i2, allocatedMax = partIndexMax;
    if (allocatedMax > CUDA_BLOCK_NUM*THREADS_PER_BLOCK) {
        allocatedMax = CUDA_BLOCK_NUM*THREADS_PER_BLOCK;
    }
    double *listCalculatedBestResultValue = new double[allocatedMax];
    double *listCalculatedDotProductBestRateValues = new double[allocatedMax*MAX_MATRIX_COLUMS];
    double *d_listCalculatedBestResultValue;
    double *d_listCalculatedDotProductBestRateValues;
    double *d_globalBestResultMax;
    int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT];
    int *d_listCounterPartIndex;
    int listCounterPartPreviousIndex[MAX_COUNTER_PART_INDEX_COUNT];
    int *d_listCounterPartPreviousIndex;
    int listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT];
    int *d_listCounterPartMathType;
    int *d_bestCounterPartMultipliersCounter;
    int *d_listResultColumnIndex;

#ifdef CUDA_COMPILE
    cudaMalloc((void**)&d_listCalculatedBestResultValue, sizeof(double)*allocatedMax);
    cudaMalloc((void**)&d_listCalculatedDotProductBestRateValues, sizeof(double)*allocatedMax*MAX_MATRIX_COLUMS);
    cudaMalloc((void**)&d_globalBestResultMax, sizeof(double));
    cudaMalloc((void**)&d_listCounterPartIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
    cudaMalloc((void**)&d_listCounterPartPreviousIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
    cudaMalloc((void**)&d_listCounterPartMathType, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
    cudaMalloc((void**)&d_bestCounterPartMultipliersCounter, sizeof(int)*MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT);
    cudaMalloc((void**)&d_listResultColumnIndex, sizeof(int)*listResultColumnIndexCount);
    cudaMemcpy(d_listResultColumnIndex, listResultColumnIndex, sizeof(int)*listResultColumnIndexCount, cudaMemcpyHostToDevice);

    const int *tmp = bestResultStorage->getCounterPartMultiplierCounter();
    cudaMemcpy(d_bestCounterPartMultipliersCounter,
                bestResultStorage->getCounterPartMultiplierCounter(),
                sizeof(int)*MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT,
                cudaMemcpyHostToDevice);
#else
    d_listCalculatedBestResultValue = new double[allocatedMax];
    d_listCalculatedDotProductBestRateValues = new double[allocatedMax*MAX_MATRIX_COLUMS];
    d_globalBestResultMax = new double;
    d_listCounterPartIndex = new int[MAX_COUNTER_PART_INDEX_COUNT];
    d_listCounterPartPreviousIndex = new int[MAX_COUNTER_PART_INDEX_COUNT];
    d_listCounterPartMathType = new int[MAX_COUNTER_PART_INDEX_COUNT];
    d_bestCounterPartMultipliersCounter = new int[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT];
    d_listResultColumnIndex = new int[listResultColumnIndexCount];
    memcpy(d_listResultColumnIndex, listResultColumnIndex, sizeof(int)*listResultColumnIndexCount);

    memcpy(d_bestCounterPartMultipliersCounter,
        bestResultStorage->getCounterPartMultiplierCounter(),
        sizeof(int)*MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT );
#endif
    memcpy(listCounterPartIndex, bestResultStorage->getListCounterPartIndex(), sizeof(listCounterPartIndex) );
    memcpy(listCounterPartPreviousIndex, bestResultStorage->getPreviousIndex(), sizeof(listCounterPartPreviousIndex) );
    for (i2=0;i2<allocatedMax;i2++) {
        listCalculatedBestResultValue[i2] = DEFAULT_BEST_VALUE;
    }

    const CounterPartMathType *mathTypeIndexList = bestResultStorage->getListCounterPartMathTypeIndex();
    for (i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        listCounterPartMathType[i] = static_cast<int>(mathTypeIndexList[i]);
    }

#ifdef CUDA_COMPILE
    cudaMemcpy(d_listCounterPartIndex, listCounterPartIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_listCounterPartPreviousIndex, listCounterPartPreviousIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_listCounterPartMathType, listCounterPartMathType, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
#else
    memcpy(d_listCounterPartIndex, listCounterPartIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
    memcpy(d_listCounterPartPreviousIndex, listCounterPartPreviousIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
    memcpy(d_listCounterPartMathType, listCounterPartMathType, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
#endif

    for (i=0;i<partIndexMax/(CUDA_BLOCK_NUM*THREADS_PER_BLOCK) + 1;i++) {
        double tmp = bestResultStorage->getCurrentBestResult();
#ifdef CUDA_COMPILE
        cudaMemcpy(d_globalBestResultMax, &tmp, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_listCalculatedBestResultValue, listCalculatedBestResultValue, sizeof(double)*allocatedMax, cudaMemcpyHostToDevice);
        studyBestResultFinetunePrimaryValue<<<CUDA_BLOCK_NUM, THREADS_PER_BLOCK>>>(d_listTableData,
                            d_listCalculatedBestResultValue,
                            d_listCalculatedDotProductBestRateValues,
                            d_globalBestResultMax,

                            d_listCounterPartIndex,
                            d_listCounterPartPreviousIndex,
                            d_listCounterPartMathType,
                            d_bestCounterPartMultipliersCounter,

                            partIndexMax, i*(CUDA_BLOCK_NUM*THREADS_PER_BLOCK),
                            d_listResultColumnIndex,
                            listResultColumnIndexCount);
        cudaDeviceSynchronize();
        cudaMemcpy(listCalculatedBestResultValue, d_listCalculatedBestResultValue, sizeof(double)*allocatedMax, cudaMemcpyDeviceToHost);
        cudaMemcpy(listCalculatedDotProductBestRateValues, d_listCalculatedDotProductBestRateValues,  sizeof(double)*allocatedMax*MAX_MATRIX_COLUMS, cudaMemcpyDeviceToHost);
#else
        memcpy(d_globalBestResultMax, &tmp, sizeof(double));
        memcpy(d_listCalculatedBestResultValue, listCalculatedBestResultValue, sizeof(double)*allocatedMax);
        for (uint64_cu partIndex=0;partIndex<CUDA_BLOCK_NUM*THREADS_PER_BLOCK;partIndex++) {
            studyBestResultFinetunePrimaryValue(d_listTableData,
                            d_listCalculatedBestResultValue,
                            d_listCalculatedDotProductBestRateValues,
                            d_globalBestResultMax,

                            d_listCounterPartIndex,
                            d_listCounterPartPreviousIndex,
                            d_listCounterPartMathType,
                            d_bestCounterPartMultipliersCounter,

                            partIndex, partIndexMax, i*(CUDA_BLOCK_NUM*THREADS_PER_BLOCK),
                            d_listResultColumnIndex,
                            listResultColumnIndexCount);
        }
        memcpy(listCalculatedBestResultValue, d_listCalculatedBestResultValue, sizeof(double)*allocatedMax);
        memcpy(listCalculatedDotProductBestRateValues, d_listCalculatedDotProductBestRateValues,  sizeof(double)*allocatedMax*MAX_MATRIX_COLUMS);
#endif
        for (i2=0;i2<allocatedMax;i2++) {
            if (bestResultStorage->getCurrentBestResult() > listCalculatedBestResultValue[i2]) {
                    bestResultStorage->setBestResultPrimaryFineTuneOnly(
                        i2 + i*(CUDA_BLOCK_NUM*THREADS_PER_BLOCK),
                        listCalculatedBestResultValue[i2], listResultColumnIndex[0],
                        listCalculatedDotProductBestRateValues+i2*MAX_MATRIX_COLUMS);
            }
            listCalculatedBestResultValue[i2] = DEFAULT_BEST_VALUE;
        }
    }
    delete[] listCalculatedBestResultValue;
    delete[] listCalculatedDotProductBestRateValues;

#ifdef CUDA_COMPILE
    cudaFree(d_listCalculatedBestResultValue);
    cudaFree(d_listCalculatedDotProductBestRateValues);
    cudaFree(d_globalBestResultMax);
    cudaFree(d_listCounterPartIndex);
    cudaFree(d_listCounterPartPreviousIndex);
    cudaFree(d_listCounterPartMathType);
    cudaFree(d_bestCounterPartMultipliersCounter);
    cudaFree(d_listResultColumnIndex);
#else
    delete[] d_listCalculatedBestResultValue;
    delete[] d_listCalculatedDotProductBestRateValues;
    delete d_globalBestResultMax;
    delete[] d_listCounterPartIndex;
    delete[] d_listCounterPartPreviousIndex;
    delete[] d_listCounterPartMathType;
    delete[] d_bestCounterPartMultipliersCounter;
    delete[] d_listResultColumnIndex;
#endif
}

#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief finetune max part index for primary value
 * for cuda calculation (how many max calculations are required with different values)
 * @return uint64_t part index max
 */
uint64_t getPartIndexMax()
{
    uint64_t ret = (PRIMARY_FINETUNE_RATE_FIND_MAX-PRIMARY_FINETUNE_RATE_FIND_MIN);

    for (size_t c=1;c<MAX_MATRIX_COLUMS;c++) {
        ret *= (PRIMARY_FINETUNE_RATE_FIND_MAX-PRIMARY_FINETUNE_RATE_FIND_MIN);
    }
    return ret;
}

}