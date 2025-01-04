/**
 * @file study_search_new_counter_part.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief searching new possible counter part 
 *
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef STUDY_SEARCH_NEW_COUNTER_PART_H
#define STUDY_SEARCH_NEW_COUNTER_PART_H

#include "../data/list_table_data.h"
#include "../define_values.h"

#ifdef CUDA_COMPILE
__host__ __device__
#endif
int getStudyBestResultNewCounterPartPositionIndex(const int *listCounterPartIndex);

#ifdef CUDA_COMPILE
__global__
#endif
void studyBestResultNewCounterPart(const ListTableData *list,
                    double *listCalculatedBestResultValue,
                    double *listCalculatedDotProductValues,
                    double *globalBestResultMax,
                    const int *bestMultipliersPrimary /* [MAX_MATRIX_COLUMS] */,
                    const int *listCounterPartIndex         /* [MAX_COUNTER_PART_INDEX_COUNT] */ ,
                    const int *listCounterPartPreviousIndex /* [MAX_COUNTER_PART_INDEX_COUNT] */ ,
                    const int *listCounterPartMathType      /* [MAX_COUNTER_PART_INDEX_COUNT] */ ,
                    const int *bestCounterPartMultipliersCounter /* [MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT] */,
                    const int *listResultColumnIndex,
                    const int listResultColumnIndexCount,
                    const uint64_cu partIndexMax, const uint64_cu partIndexAdd
#ifndef CUDA_COMPILE
                    , uint64_cu partIndex
#endif
                    );

#endif // STUDY_SEARCH_NEW_COUNTER_PART_H
