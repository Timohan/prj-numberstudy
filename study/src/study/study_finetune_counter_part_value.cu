/**
 * @file study_finetune_counter_part_value.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief finetune counter part
 * 
 * @copyright Copyright (c) 2024
 */
#include "study_finetune_counter_part_value.h"
#ifndef CUDA_COMPILE
#include <cstdlib>
#include <cstring>
#else
#include <cuda.h>
#endif
#include "../data/list_table_data.h"
#include "../study_cuda/study_best_result_finetune_counter_part_value.h"
#include "../loader/best_result_storage.h"
#include <stdio.h>
#include "../macros.h"

namespace StudyFinetuneCounterPartValue
{
#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief finetune counter part search
 * 
 * @param d_listTableData pointer to list table data in gpu
 * @param listResultColumnIndex list of columns to find new counter part
 * @param listResultColumnIndexCount list of columns count
 * @param bestResultStorage current best result
 * @param counterPartIndex counter part index to finetune
 */
void study(ListTableData *d_listTableData,
           const int *listResultColumnIndex,
           const int listResultColumnIndexCount,
           BestResultStorage *bestResultStorage,
           const unsigned int counterPartIndex)
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
    int *d_bestMultipliersPrimary;
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
    PRJ_MEMCPY(d_bestCounterPartMultipliersCounter,
                bestResultStorage->getCounterPartMultiplierCounter(),
                sizeof(int)*MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT,
                cudaMemcpyHostToDevice);
    PRJ_MEMCPY(d_bestMultipliersPrimary, bestResultStorage->getCounterPartMultiplierPrimary(), sizeof(int)*MAX_MATRIX_COLUMS, cudaMemcpyHostToDevice);

    memcpy(listCounterPartIndex, bestResultStorage->getListCounterPartIndex(), sizeof(listCounterPartIndex) );
    memcpy(listCounterPartPreviousIndex, bestResultStorage->getPreviousIndex(), sizeof(listCounterPartPreviousIndex) );
    for (i2=0;i2<allocatedMax;i2++) {
        listCalculatedBestResultValue[i2] = DEFAULT_BEST_VALUE;
    }

    const CounterPartMathType *mathTypeIndexList = bestResultStorage->getListCounterPartMathTypeIndex();
    for (i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        listCounterPartMathType[i] = static_cast<int>(mathTypeIndexList[i]);
    }

    PRJ_MEMCPY(d_listCounterPartIndex, listCounterPartIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
    PRJ_MEMCPY(d_listCounterPartPreviousIndex, listCounterPartPreviousIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);
    PRJ_MEMCPY(d_listCounterPartMathType, listCounterPartMathType, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT, cudaMemcpyHostToDevice);

    for (i=0;i<partIndexMax/(CUDA_BLOCK_NUM*THREADS_PER_BLOCK) + 1;i++) {
        double tmp = bestResultStorage->getCurrentBestResult();
        PRJ_MEMCPY(d_globalBestResultMax, &tmp, sizeof(double), cudaMemcpyHostToDevice);
        PRJ_MEMCPY(d_listCalculatedBestResultValue, listCalculatedBestResultValue, sizeof(double)*allocatedMax, cudaMemcpyHostToDevice);
        PRJ_FUNC_CALL(studyBestResultFinetuneCounterPartValue, d_listTableData,
                            d_listCalculatedBestResultValue,
                            d_listCalculatedDotProductBestRateValues,
                            d_globalBestResultMax,
                            d_bestMultipliersPrimary,

                            d_listCounterPartIndex,
                            d_listCounterPartPreviousIndex,
                            d_listCounterPartMathType,
                            d_bestCounterPartMultipliersCounter,

                            counterPartIndex,
                            d_listResultColumnIndex,
                            listResultColumnIndexCount,
                            partIndexMax, i*(CUDA_BLOCK_NUM*THREADS_PER_BLOCK));
        PRJ_CUDA_WAIT();
        PRJ_MEMCPY(listCalculatedBestResultValue, d_listCalculatedBestResultValue, sizeof(double)*allocatedMax, cudaMemcpyDeviceToHost);
        PRJ_MEMCPY(listCalculatedDotProductBestRateValues, d_listCalculatedDotProductBestRateValues, sizeof(double)*allocatedMax*MAX_MATRIX_COLUMS, cudaMemcpyDeviceToHost);
        for (i2=0;i2<allocatedMax;i2++) {
            if (bestResultStorage->getCurrentBestResult() > listCalculatedBestResultValue[i2]) {
                    bestResultStorage->setBestResultCounterPartFineTuneOnly(
                        counterPartIndex,
                        i2 + i*(CUDA_BLOCK_NUM*THREADS_PER_BLOCK),
                        listCalculatedBestResultValue[i2], listResultColumnIndex[0],
                        listCalculatedDotProductBestRateValues+i2*MAX_MATRIX_COLUMS);
            }
            listCalculatedBestResultValue[i2] = DEFAULT_BEST_VALUE;
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
}


#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief finetune counter parts
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
    for (unsigned int counterPartIndex=0;counterPartIndex<MAX_COUNTER_PART_INDEX_COUNT;counterPartIndex++) {
        int *listCounterPart = bestResultStorage->getListCounterPartIndex();
        if (listCounterPart[counterPartIndex] == 0) {
            continue;
        }
        printf("Study searching counter part value finetunes %d of %d\n",
                counterPartIndex+1, bestResultStorage->getListCounterPartCount()+1);
        study(d_listTableData, listResultColumnIndex,
              listResultColumnIndexCount, bestResultStorage,
              counterPartIndex);
    }
}

#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief finetune max part index for counter parts
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
