/**
 * @file study_best_result_finetune_counter_part_value.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief finetune counter part
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "study_best_result_finetune_counter_part_value.h"
#include "../define_values.h"
#include "../search_part_index/search_part_index.h"
#include "study_best_result.h"
#include <stdio.h>

#ifndef CUDA_COMPILE
#include <cstring>
#endif

#ifdef CUDA_COMPILE
__global__
void studyBestResultFinetuneCounterPartValue(const ListTableData *list,
                    double *listCalculatedBestResultValue,
                    double *listCalculatedDotProductValues,
                    double *globalBestResultMax,
                    const int *bestMultipliersPrimary /* [MAX_MATRIX_COLUMS] */,
                    const int *listCounterPartIndex         /* [MAX_COUNTER_PART_INDEX_COUNT] */ ,
                    const int *listCounterPartPreviousIndex /* [MAX_COUNTER_PART_INDEX_COUNT] */ ,
                    const int *listCounterPartMathType      /* [MAX_COUNTER_PART_INDEX_COUNT] */ ,
                    const int *bestCounterPartMultipliersCounter /* [MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT] */,
                    const unsigned int counterPartIndex,
                    const uint64_cu partIndexMax, const uint64_cu partIndexAdd,
                    const int *listResultColumnIndex,
                    const int listResultColumnIndexCount)
#else
/**
 * @brief study finetune counter part values
 * 
 * @param list table data list
 * @param listCalculatedBestResultValue [out] calculated best result value by this function for this part index
 * @param listCalculatedDotProductValues [out] calculated best dot products by this function for this part index
 * @param globalBestResultMax [in/out] current best global result
 * @param bestMultipliersPrimary current best multipliers for primary table
 * @param listCounterPartIndex current found best counter parts
 * @param listCounterPartPreviousIndex current found best previous indexes for counter parts
 * @param listCounterPartMathType current found best math types for counter parts
 * @param bestCounterPartMultipliersCounter current best multpliers for counter parts
 * @param counterPartIndex new possible counter part index
 * @param partIndex part index to calculate new multpliers for counterPartIndex (on gpu, this is calculated from blockIdx)
 * @param partIndexMax max part index
 * @param partIndexAdd value to add to get real part index
 * @param listResultColumnIndex list of result columns that are used to study
 * @param listResultColumnIndexCount list of result columns count
 */
void studyBestResultFinetuneCounterPartValue(const ListTableData *list,
                    double *listCalculatedBestResultValue,
                    double *listCalculatedDotProductValues,
                    double *globalBestResultMax,
                    const int *bestMultipliersPrimary /* [MAX_MATRIX_COLUMS] */,
                    const int *listCounterPartIndex         /* [MAX_COUNTER_PART_INDEX_COUNT] */ ,
                    const int *listCounterPartPreviousIndex /* [MAX_COUNTER_PART_INDEX_COUNT] */ ,
                    const int *listCounterPartMathType      /* [MAX_COUNTER_PART_INDEX_COUNT] */ ,
                    const int *bestCounterPartMultipliersCounter /* [MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT] */,
                    const unsigned int counterPartIndex,
                    uint64_cu partIndex, const uint64_cu partIndexMax, uint64_cu partIndexAdd,
                    const int *listResultColumnIndex,
                    const int listResultColumnIndexCount)
#endif
{
#ifdef CUDA_COMPILE
    uint64_cu partIndex = static_cast<uint64_cu>(blockDim.x * blockIdx.x + threadIdx.x) + partIndexAdd;
#else
    partIndex += partIndexAdd;
#endif
    int i;
    double localBestResult = DEFAULT_BEST_VALUE;
    double calculatedDotProductValues[MAX_MATRIX_COLUMS];
    CounterPartMathType listCounterPartMathTypeEnum[MAX_COUNTER_PART_INDEX_COUNT];
    int multiplier[MAX_MATRIX_COLUMS];
    int counterPartMultipliersCounter[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT];

    if (partIndex >= partIndexMax) {
        return;
    }

    for (i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        listCounterPartMathTypeEnum[i] = static_cast<CounterPartMathType>(listCounterPartMathType[i]);
    }

    generateRatesPrimaryFineTuneValue(partIndex, multiplier);
    memcpy(counterPartMultipliersCounter, bestCounterPartMultipliersCounter, sizeof(counterPartMultipliersCounter));
    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        counterPartMultipliersCounter[i+MAX_MATRIX_COLUMS*counterPartIndex] = multiplier[i];
    }
    if (studyBestResult(globalBestResultMax,
                        &localBestResult,
                        calculatedDotProductValues,
                        list,
                        bestMultipliersPrimary,
                        counterPartMultipliersCounter,
                        listCounterPartPreviousIndex,
                        listResultColumnIndex,
                        listResultColumnIndexCount,
                        listCounterPartIndex,
                        listCounterPartMathTypeEnum)) {
        if (listCalculatedBestResultValue[ (partIndex - partIndexAdd) ] > localBestResult) {
            listCalculatedBestResultValue[ (partIndex - partIndexAdd) ] = localBestResult;
            for (i=0;i<MAX_MATRIX_COLUMS;i++) {
                listCalculatedDotProductValues[ static_cast<int>((partIndex - partIndexAdd)*MAX_MATRIX_COLUMS) + i ] = calculatedDotProductValues[i];
            }
        }
    }
}