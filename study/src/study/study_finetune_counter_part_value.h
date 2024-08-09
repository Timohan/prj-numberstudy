/**
 * @file study_finetune_counter_part_value.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief finetune counter part
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef STUDY_FINETUNE_COUNTER_PART_VALUE_H
#define STUDY_FINETUNE_COUNTER_PART_VALUE_H

#include "../define_values.h"
#include <stdint.h>

struct ListTableData;
class BestResultStorage;

namespace StudyFinetuneCounterPartValue
{
#ifdef CUDA_COMPILE
__host__
#endif
void study(ListTableData *d_listTableData, 
           const int *listResultColumnIndex,
           const int listResultColumnIndexCount,
           BestResultStorage *bestResultStorage);

#ifdef CUDA_COMPILE
__host__
#endif
void study(ListTableData *d_listTableData,
           const int *listResultColumnIndex,
           const int listResultColumnIndexCount,
           BestResultStorage *bestResultStorage,
           const unsigned int counterPartIndex);

uint64_t getPartIndexMax();
};


#endif // STUDY_FINETUNE_COUNTER_PART_VALUE_H
