/**
 * @file study_best_result_finetune_counter_part_value.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief finetune counter part
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef STUDY_BEST_RESULT_FINETUNE_COUNTER_PART_VALUE_H
#define STUDY_BEST_RESULT_FINETUNE_COUNTER_PART_VALUE_H

#include "../data/list_table_data.h"

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
                    const int listResultColumnIndexCount);
#else
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
                    const int listResultColumnIndexCount);
#endif

#endif // STUDY_BEST_RESULT_FINETUNE_COUNTER_PART_VALUE_H
