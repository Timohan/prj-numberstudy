/**
 * @file search_part_index.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief search multipliers from part index
 * 
 * @copyright Copyright (c) 2024
 */
#include "search_part_index.h"

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief finds multipliers from part index
 * 
 * @param partOut [out] multiplier values will be filled here
 * @param partDivider dividers list to calculate multilier
 * @param partMultiplierIndex index of partOut
 * @param partIndex multiplier is calculated from this
 */
void searchPartIndex(int *partOut, const int *partDivider, int partMultiplierIndex, uint64_cu partIndex)
{
    if (partMultiplierIndex <= 1) {
        partOut[0] = partIndex;
        return;
    }
    int i;
    uint64_cu divider = static_cast<uint64_cu>(partDivider[0]);
    for (i=1;i<partMultiplierIndex-1;i++) {
        divider *= static_cast<uint64_cu>(partDivider[i]);
    }
    partOut[partMultiplierIndex-1] = static_cast<int>(partIndex/divider);
    searchPartIndex(partOut, partDivider, partMultiplierIndex-1,
        partIndex - static_cast<uint64_cu>(partOut[partMultiplierIndex-1]*divider) );
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief generates primary multiplier values from part index value
 * 
 * @param partIndex 
 * @param multiplier [out] primary multipliers from part index
 */
void generateRatesPrimaryValue(uint64_cu partIndex, int multiplier[MAX_MATRIX_COLUMS])
{
    int i, ratesIndex = 0;
    int ratesMultipliers[MAX_MATRIX_COLUMS];

    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        ratesMultipliers[ratesIndex] = PRIMARY_RATE_MULTIPLIER_STEP_COUNT;
        ratesIndex++;
    }

    searchPartIndex(multiplier, ratesMultipliers, ratesIndex, partIndex);

    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        multiplier[i] = (multiplier[i]+1)*100/PRIMARY_RATE_MULTIPLIER_STEP_COUNT;
    }
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief generates counter part multiplier values from part index value
 * 
 * @param partIndex 
 * @param multiplier [out] primary multipliers from part index
 */
void generateRatesCounterPartValue(uint64_cu partIndex, int multiplier[MAX_MATRIX_COLUMS])
{
    int i, ratesIndex = 0;
    int ratesMultipliers[MAX_MATRIX_COLUMS];

    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        ratesMultipliers[ratesIndex] = COUNTER_PART_RATE_MULTIPLIER_STEP_COUNT;
        ratesIndex++;
    }

    searchPartIndex(multiplier, ratesMultipliers, ratesIndex, partIndex);

    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        multiplier[i] = (multiplier[i]+1)*100/COUNTER_PART_RATE_MULTIPLIER_STEP_COUNT;
    }
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief generates primary finetune values from part index value
 * 
 * @param partIndex 
 * @param multiplier [out] multipliers from part index
 */
void generateRatesPrimaryFineTuneValue(uint64_cu partIndex, int multiplier[MAX_MATRIX_COLUMS])
{
    int i, ratesIndex = 0;
    int ratesMultipliers[MAX_MATRIX_COLUMS];

    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        ratesMultipliers[ratesIndex] = (PRIMARY_FINETUNE_RATE_FIND_MAX-PRIMARY_FINETUNE_RATE_FIND_MIN);
        ratesIndex++;
    }

    searchPartIndex(multiplier, ratesMultipliers, ratesIndex, partIndex);

    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        multiplier[i] = multiplier[i] + PRIMARY_FINETUNE_RATE_FIND_MIN;
    }
}