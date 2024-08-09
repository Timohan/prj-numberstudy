/**
 * @file study_best_result.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief general calculate/study start dot products and result value
 *
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef STUDY_BEST_RESULT_H
#define STUDY_BEST_RESULT_H

#include "../data/list_table_data.h"
#include "../define_values.h"

#ifdef CUDA_COMPILE
__host__ __device__
#endif
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
                    const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT]);
#endif // STUDY_SHORT_RESULT_H
