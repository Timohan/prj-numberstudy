/**
 * @file study_find_new_counter_part.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief finds new counter part
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef STUDY_FIND_NEW_COUNTER_PART_H
#define STUDY_FIND_NEW_COUNTER_PART_H

#include "../define_values.h"
#include <stdint.h>

struct ListTableData;
class BestResultStorage;

namespace StudyFindNewCounterPart
{
#ifdef CUDA_COMPILE
__host__
#endif
bool study(ListTableData *d_listTableData, const int *listResultColumnIndex,
           const int listResultColumnIndexCount,
           BestResultStorage *bestResultStorage, const unsigned int counterPartCountMax);

#ifdef CUDA_COMPILE
__host__
#endif
uint64_t getPartIndexMax();

};

#endif // STUDY_FIND_NEW_COUNTER_PART_H
