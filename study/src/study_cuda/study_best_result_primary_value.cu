/**
 * @file study_best_result_primary_value.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief calculate/study start dot products and result value
 *
 * @copyright Copyright (c) 2024
 * 
 */
#include "study_best_result_primary_value.h"
#include "../define_values.h"
#include "../search_part_index/search_part_index.h"
#include "study_best_result.h"
#include <stdio.h>
#ifndef CUDA_COMPILE
#include <cstring>
#endif

#ifdef CUDA_COMPILE
__global__
void studyBestResultPrimaryValue(const ListTableData *list,
                    double *listCalculatedBestResultValue,
                    double *listCalculatedBestRateValues,
                    double *globalBestResultMax,
                    const int *listResultColumnIndex,
                    const int listResultColumnIndexCount)
#else
/**
 * @brief study best primary value
 * 
 * @param list list table data
 * @param listCalculatedBestResultValue [out] calculated best result value
 * @param listCalculatedBestRateValues [out] calculated best rate result value
 * @param globalBestResultMax [out] calculated best result
 * @param listResultColumnIndex list of column indexes to calculate best result value
 * @param listResultColumnIndexCount list of column indexes count
 */
void studyBestResultPrimaryValue(const ListTableData *list,
                    double *listCalculatedBestResultValue,
                    double *listCalculatedBestRateValues,
                    double *globalBestResultMax,
                    const int *listResultColumnIndex,
                    const int listResultColumnIndexCount)
#endif
{
    int i;
    double localBestResult = DEFAULT_BEST_VALUE;
    int bestCounterPartMultipliersPrimary[MAX_MATRIX_COLUMS];
    int bestCounterPartMultipliersCounter[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT];
    double calculatedBestRateValues[MAX_MATRIX_COLUMS];
    CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT];
    int previousIndex[MAX_COUNTER_PART_INDEX_COUNT];
    int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT];

    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        bestCounterPartMultipliersPrimary[i] = DEFAULT_PRIMARY_VALUE_MULTIPLIER;
    }
    memset(bestCounterPartMultipliersCounter, 0, sizeof(bestCounterPartMultipliersCounter));
    memset(previousIndex, 0, sizeof(previousIndex));
    memset(listCounterPartIndex, 0, sizeof(listCounterPartIndex));
    memset(listCounterPartMathType, 0, sizeof(listCounterPartMathType));

    if (studyBestResult(globalBestResultMax,
                        &localBestResult,
                        calculatedBestRateValues,
                        list,
                        bestCounterPartMultipliersPrimary,
                        bestCounterPartMultipliersCounter,
                        previousIndex,
                        listResultColumnIndex,
                        listResultColumnIndexCount,
                        listCounterPartIndex,
                        listCounterPartMathType)) {
        if (listCalculatedBestResultValue[ 0 ] > localBestResult) {
            listCalculatedBestResultValue[ 0 ] = localBestResult;
            for (i=0;i<MAX_MATRIX_COLUMS;i++) {
                listCalculatedBestRateValues[i] = calculatedBestRateValues[i];
            }
        }
    }
}