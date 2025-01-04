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
#include "../macros.h"
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

    PRJ_ALLOC(d_listCalculatedBestResultValue, double, allocatedMax);
    PRJ_ALLOC(d_listCalculatedDotProductBestRateValues, double, allocatedMax*MAX_MATRIX_COLUMS);
    PRJ_ALLOC(d_globalBestResultMax, double, 1);
    PRJ_ALLOC(d_bestMultipliersPrimary, int, MAX_MATRIX_COLUMS);
    PRJ_ALLOC(d_listCounterPartIndex, int, MAX_COUNTER_PART_INDEX_COUNT);
    PRJ_ALLOC(d_listCounterPartPreviousIndex, int, MAX_COUNTER_PART_INDEX_COUNT);
    PRJ_ALLOC(d_listCounterPartMathType, int, MAX_COUNTER_PART_INDEX_COUNT);
    PRJ_ALLOC(d_bestCounterPartMultipliersCounter, int, MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT);
    PRJ_ALLOC(d_listResultColumnIndex, int, listResultColumnIndexCount);
    PRJ_MEMCPY(d_listResultColumnIndex, listResultColumnIndex, sizeof(int)*listResultColumnIndexCount, cudaMemcpyHostToDevice);
    PRJ_MEMCPY(d_bestMultipliersPrimary, bestResultStorage->getCounterPartMultiplierPrimary(), sizeof(int)*MAX_MATRIX_COLUMS, cudaMemcpyHostToDevice);
    PRJ_MEMCPY(d_bestCounterPartMultipliersCounter, bestResultStorage->getCounterPartMultiplierCounter(), sizeof(int)*MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);

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
        PRJ_MEMCPY(d_listCounterPartIndex, listCounterPartIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
        for (newPreviousIndex=MIN_PREVIOUS_NEXT_INDEX;newPreviousIndex<=MAX_PREVIOUS_NEXT_INDEX;newPreviousIndex++) {
            listCounterPartPreviousIndex[newCounterPartPositionIndex] = newPreviousIndex;
            PRJ_MEMCPY(d_listCounterPartPreviousIndex, listCounterPartPreviousIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
            for (newCounterParthMathTypeIndex=0;newCounterParthMathTypeIndex<static_cast<int>(CounterPartMathType::CounterPartMathType_Count);newCounterParthMathTypeIndex++) {
                listCounterPartMathType[newCounterPartPositionIndex] = newCounterParthMathTypeIndex;
                PRJ_MEMCPY(d_listCounterPartMathType, listCounterPartMathType, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
                for (i=0;i<partIndexMax/(CUDA_BLOCK_NUM*THREADS_PER_BLOCK) + 1;i++) {
                    double tmp = bestResultStorage->getCurrentBestResult();
                    PRJ_MEMCPY(d_globalBestResultMax, &tmp, sizeof(double), cudaMemcpyHostToDevice);
                    PRJ_MEMCPY(d_listCalculatedBestResultValue, listCalculatedBestResultValue, sizeof(double)*allocatedMax, cudaMemcpyHostToDevice);
                    PRJ_FUNC_CALL(studyBestResultNewCounterPart, d_listTableData,
                            d_listCalculatedBestResultValue,
                            d_listCalculatedDotProductBestRateValues,
                            d_globalBestResultMax,
                            d_bestMultipliersPrimary,

                            d_listCounterPartIndex,
                            d_listCounterPartPreviousIndex,
                            d_listCounterPartMathType,
                            d_bestCounterPartMultipliersCounter,

                            d_listResultColumnIndex,
                            listResultColumnIndexCount,
                            partIndexMax, i*(CUDA_BLOCK_NUM*THREADS_PER_BLOCK));
                    PRJ_CUDA_WAIT();
                    PRJ_MEMCPY(listCalculatedBestResultValue, d_listCalculatedBestResultValue, sizeof(double)*allocatedMax, cudaMemcpyDeviceToHost);
                    PRJ_MEMCPY(listCalculatedDotProductBestRateValues, d_listCalculatedDotProductBestRateValues, sizeof(double)*allocatedMax*MAX_MATRIX_COLUMS, cudaMemcpyDeviceToHost);
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
    PRJ_FREE(d_listCalculatedBestResultValue)
    PRJ_FREE(d_listCalculatedDotProductBestRateValues)
    PRJ_FREE(d_globalBestResultMax)
    PRJ_FREE(d_bestMultipliersPrimary)
    PRJ_FREE(d_listCounterPartIndex)
    PRJ_FREE(d_listCounterPartPreviousIndex)
    PRJ_FREE(d_listCounterPartMathType)
    PRJ_FREE(d_bestCounterPartMultipliersCounter)
    PRJ_FREE(d_listResultColumnIndex)
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