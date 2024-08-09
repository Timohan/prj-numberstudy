/**
 * @file search_part_index.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief search multipliers from part index
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef SEARCH_PART_INDEX_H
#define SEARCH_PART_INDEX_H

#include "../define_values.h"

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void searchPartIndex(int *partOut, const int *partDivider, int partMultiplierIndex, uint64_cu partIndex);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void generateRatesPrimaryValue(uint64_cu partIndex, int multiplier[MAX_MATRIX_COLUMS]);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void generateRatesCounterPartValue(uint64_cu partIndex, int multiplier[MAX_MATRIX_COLUMS]);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void generateRatesPrimaryFineTuneValue(uint64_cu partIndex, int multiplier[MAX_MATRIX_COLUMS]);

#endif // SEARCH_PART_INDEX_H
