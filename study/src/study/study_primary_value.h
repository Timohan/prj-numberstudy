/**
 * @file study_primary_value.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief calculate primary value's start values
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef STUDY_PRIMARY_VALUE_H
#define STUDY_PRIMARY_VALUE_H

#include "../define_values.h"
#include <stdint.h>

struct ListTableData;
class BestResultStorage;

namespace StudyPrimaryValue
{
#ifdef CUDA_COMPILE
__host__
#endif
void study(ListTableData *d_listTableData,
           const int *listResultColumnIndex,
           const int listResultColumnIndexCount,
           BestResultStorage *bestResultStorage);
};

#endif // STUDY_PRIMARY_VALUE_H
