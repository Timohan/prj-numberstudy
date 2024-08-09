/**
 * @file study_best_result_primary_value.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief calculate/study start dot products and result value
 *
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef STUDY_BEST_RESULT_PRIMARY_VALUE_H
#define STUDY_BEST_RESULT_PRIMARY_VALUE_H

#include "../data/list_table_data.h"

#ifdef CUDA_COMPILE
__global__
void studyBestResultPrimaryValue(const ListTableData *list,
                    double *listCalculatedBestResultValue,
                    double *listCalculatedBestRateValues,
                    double *globalBestResultMax,
                    const int *listResultColumnIndex,
                    const int listResultColumnIndexCount);
#else
void studyBestResultPrimaryValue(const ListTableData *list,
                    double *listCalculatedBestResultValue,
                    double *listCalculatedBestRateValues,
                    double *globalBestResultMax,
                    const int *listResultColumnIndex,
                    const int listResultColumnIndexCount);
#endif

#endif // STUDY_BEST_RESULT_PRIMARY_VALUE_H
