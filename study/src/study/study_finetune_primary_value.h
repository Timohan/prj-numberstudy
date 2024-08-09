/**
 * @file study_finetune_primary_value.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief finetune primary value search
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef STUDY_FINETUNE_PRIMARY_VALUE_H
#define STUDY_FINETUNE_PRIMARY_VALUE_H

#include "../define_values.h"
#include <stdint.h>

struct ListTableData;
class BestResultStorage;

namespace StudyFinetunePrimaryValue
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
uint64_t getPartIndexMax();
};


#endif // STUDY_FINETUNE_PRIMARY_VALUE_H
