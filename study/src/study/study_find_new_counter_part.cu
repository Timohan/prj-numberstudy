/**
 * @file study_find_new_counter_part.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief finds new counter part
 * 
 * @copyright Copyright (c) 2024
 */
#include "study_find_new_counter_part.h"
#include "../study_cuda/study_search_new_counter_part.h"
#include "../loader/best_result_storage.h"
#include <stdio.h>

#ifndef CUDA_COMPILE
#include <cstring>
#endif

namespace StudyFindNewCounterPart
{
#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief cpu side study find new counter part
 * 
 * @param d_listTableData pointer to gpu list table data
 * @param listResultColumnIndex list of columns to find new counter part
 * @param listResultColumnIndexCount list of columns count
 * @param bestResultStorage current best result
 * @param counterPartCountMax counter part max count
 * @return true if new counter part was found.
 * @return false if not
 */
bool study(ListTableData *d_listTableData,
           const int *listResultColumnIndex,
           const int listResultColumnIndexCount,
           BestResultStorage *bestResultStorage,
           const unsigned int counterPartCountMax)
{
    bool ret = false;
    unsigned int counterPartIndex;
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
    int *d_bestMultipliersPrimary;
    int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT];
    int *d_listCounterPartIndex;
    int listCounterPartPreviousIndex[MAX_COUNTER_PART_INDEX_COUNT];
    int *d_listCounterPartPreviousIndex;
    int newPreviousIndex;
    int listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT];
    int *d_listCounterPartMathType;
    int newCounterParthMathTypeIndex;
    int *d_bestCounterPartMultipliersCounter;
    int *d_listResultColumnIndex;

#ifdef CUDA_COMPILE
    cudaMalloc((void**)&d_listCalculatedBestResultValue, sizeof(double)*allocatedMax);
    cudaMalloc((void**)&d_listCalculatedDotProductBestRateValues, sizeof(double)*allocatedMax*MAX_MATRIX_COLUMS);
    cudaMalloc((void**)&d_globalBestResultMax, sizeof(double));
    cudaMalloc((void**)&d_bestMultipliersPrimary, sizeof(int)*MAX_MATRIX_COLUMS);
    cudaMalloc((void**)&d_listCounterPartIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
    cudaMalloc((void**)&d_listCounterPartPreviousIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
    cudaMalloc((void**)&d_listCounterPartMathType, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
    cudaMalloc((void**)&d_bestCounterPartMultipliersCounter, sizeof(int)*MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT);
    cudaMalloc((void**)&d_listResultColumnIndex, sizeof(int)*listResultColumnIndexCount);
    cudaMemcpy(d_listResultColumnIndex, listResultColumnIndex, sizeof(int)*listResultColumnIndexCount, cudaMemcpyHostToDevice);

    cudaMemcpy(d_bestMultipliersPrimary, bestResultStorage->getCounterPartMultiplierPrimary(), sizeof(int)*MAX_MATRIX_COLUMS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bestCounterPartMultipliersCounter, bestResultStorage->getCounterPartMultiplierCounter(), sizeof(int)*MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
#else
    d_listCalculatedBestResultValue = new double[allocatedMax];
    d_listCalculatedDotProductBestRateValues = new double[allocatedMax*MAX_MATRIX_COLUMS];
    d_globalBestResultMax = new double;
    d_bestMultipliersPrimary = new int[MAX_MATRIX_COLUMS];
    d_listCounterPartIndex = new int[MAX_COUNTER_PART_INDEX_COUNT];
    d_listCounterPartPreviousIndex = new int[MAX_COUNTER_PART_INDEX_COUNT];
    d_listCounterPartMathType = new int[MAX_COUNTER_PART_INDEX_COUNT];
    d_bestCounterPartMultipliersCounter = new int[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT];
    d_listResultColumnIndex = new int[listResultColumnIndexCount];
    memcpy(d_listResultColumnIndex, listResultColumnIndex, sizeof(int)*listResultColumnIndexCount);

    memcpy(d_bestMultipliersPrimary, bestResultStorage->getCounterPartMultiplierPrimary(), sizeof(int)*MAX_MATRIX_COLUMS );
    memcpy(d_bestCounterPartMultipliersCounter, bestResultStorage->getCounterPartMultiplierCounter(),  sizeof(int)*MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT );
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

    int newCounterPartPositionIndex = getStudyBestResultNewCounterPartPositionIndex(listCounterPartIndex) + 1;

    for (counterPartIndex=1;counterPartIndex<counterPartCountMax;counterPartIndex++) {
        listCounterPartIndex[newCounterPartPositionIndex] = counterPartIndex;
#ifdef CUDA_COMPILE
        cudaMemcpy(d_listCounterPartIndex, listCounterPartIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
#else
        memcpy(d_listCounterPartIndex, listCounterPartIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
#endif

        for (newPreviousIndex=MIN_PREVIOUS_NEXT_INDEX;newPreviousIndex<=MAX_PREVIOUS_NEXT_INDEX;newPreviousIndex++) {
            listCounterPartPreviousIndex[newCounterPartPositionIndex] = newPreviousIndex;
#ifdef CUDA_COMPILE
            cudaMemcpy(d_listCounterPartPreviousIndex, listCounterPartPreviousIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
#else
            memcpy(d_listCounterPartPreviousIndex, listCounterPartPreviousIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
#endif
            for (newCounterParthMathTypeIndex=0;newCounterParthMathTypeIndex<static_cast<int>(CounterPartMathType::CounterPartMathType_Count);newCounterParthMathTypeIndex++) {
                listCounterPartMathType[newCounterPartPositionIndex] = newCounterParthMathTypeIndex;
#ifdef CUDA_COMPILE
                cudaMemcpy(d_listCounterPartMathType, listCounterPartMathType, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
#else
                memcpy(d_listCounterPartMathType, listCounterPartMathType, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
#endif
                for (i=0;i<partIndexMax/(CUDA_BLOCK_NUM*THREADS_PER_BLOCK) + 1;i++) {
                    double tmp = bestResultStorage->getCurrentBestResult();
#ifdef CUDA_COMPILE
                    cudaMemcpy(d_globalBestResultMax, &tmp, sizeof(double), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_listCalculatedBestResultValue, listCalculatedBestResultValue, sizeof(double) * allocatedMax, cudaMemcpyHostToDevice);
                    studyBestResultNewCounterPart<<<CUDA_BLOCK_NUM, THREADS_PER_BLOCK>>>(d_listTableData,
                            d_listCalculatedBestResultValue,
                            d_listCalculatedDotProductBestRateValues,
                            d_globalBestResultMax,
                            d_bestMultipliersPrimary,

                            d_listCounterPartIndex,
                            d_listCounterPartPreviousIndex,
                            d_listCounterPartMathType,
                            d_bestCounterPartMultipliersCounter,

                            partIndexMax, i*(CUDA_BLOCK_NUM*THREADS_PER_BLOCK),
                            d_listResultColumnIndex,
                            listResultColumnIndexCount);
                    cudaDeviceSynchronize();
                    cudaMemcpy(listCalculatedBestResultValue, d_listCalculatedBestResultValue, sizeof(double) * allocatedMax, cudaMemcpyDeviceToHost);
                    cudaMemcpy(listCalculatedDotProductBestRateValues, d_listCalculatedDotProductBestRateValues,  sizeof(double)*allocatedMax*MAX_MATRIX_COLUMS, cudaMemcpyDeviceToHost);
#else
                    memcpy(d_globalBestResultMax, &tmp, sizeof(double));
                    memcpy(d_listCalculatedBestResultValue, listCalculatedBestResultValue, sizeof(double)*allocatedMax);
                    for (uint64_cu partIndex=0;partIndex<CUDA_BLOCK_NUM*THREADS_PER_BLOCK;partIndex++) {
                        studyBestResultNewCounterPart(d_listTableData,
                            d_listCalculatedBestResultValue,
                            d_listCalculatedDotProductBestRateValues,
                            d_globalBestResultMax,
                            d_bestMultipliersPrimary,

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
                            bestResultStorage->setBestResultNewCounterPart(
                                i2 + i*(CUDA_BLOCK_NUM*THREADS_PER_BLOCK),
                                listCalculatedBestResultValue[i2], listResultColumnIndex[0],
                                listCalculatedDotProductBestRateValues+i2*MAX_MATRIX_COLUMS,
                                newCounterPartPositionIndex,
                                counterPartIndex,
                                newPreviousIndex,
                                static_cast<CounterPartMathType>(newCounterParthMathTypeIndex));
                                ret = true;
                        }
                        listCalculatedBestResultValue[i2] = DEFAULT_BEST_VALUE;
                    }
                }
            }
        }
    }
    delete[] listCalculatedBestResultValue;
    delete[] listCalculatedDotProductBestRateValues;

#ifdef CUDA_COMPILE
    cudaFree(d_listCalculatedBestResultValue);
    cudaFree(d_listCalculatedDotProductBestRateValues);
    cudaFree(d_globalBestResultMax);
    cudaFree(d_bestMultipliersPrimary);
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
    delete[] d_bestMultipliersPrimary;
    delete[] d_listCounterPartPreviousIndex;
    delete[] d_listCounterPartMathType;
    delete[] d_bestCounterPartMultipliersCounter;
    delete[] d_listResultColumnIndex;
#endif
    return ret;
}

#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief calculates max part index for counter parts
 * for cuda calculation (how many max calculations are required with different values)
 * @return uint64_t part index max
 */
uint64_t getPartIndexMax()
{
    uint64_t ret = COUNTER_PART_RATE_MULTIPLIER_STEP_COUNT;

    for (size_t c=1;c<MAX_MATRIX_COLUMS;c++) {
        ret *= COUNTER_PART_RATE_MULTIPLIER_STEP_COUNT;
    }
    return ret;
}

}