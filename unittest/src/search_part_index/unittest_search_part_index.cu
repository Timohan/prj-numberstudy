#include "unittest_search_part_index.h"
#include <stdio.h>
#ifndef CUDA_COMPILE
#include <cstdlib>
#include <cstring>
#else
#include <cuda.h>
#endif
#include "../../../study/src/search_part_index/search_part_index.h"

#ifdef CUDA_COMPILE
__host__
#endif
bool UnitSearchPartIndex::search()
{
    int i;
    int multiplayer[MAX_MATRIX_COLUMS];
    int maxValueMultiPlayer[MAX_MATRIX_COLUMS];
    uint64_cu r = PRIMARY_RATE_MULTIPLIER_STEP_COUNT;
    uint64_cu partIndex;

    memset(maxValueMultiPlayer, 0, sizeof(maxValueMultiPlayer));
    for (partIndex=0;partIndex<r*r*r*r;partIndex++) {
        memset(multiplayer, -1, sizeof(multiplayer));
        generateRatesPrimaryValue(partIndex, multiplayer);
        if (!isValidRates(multiplayer)) {
            printf("%s (%d):invalid rate: %llu\n", __FUNCTION__, __LINE__, partIndex);
            return false;
        }
        setMaxValue(multiplayer, maxValueMultiPlayer);
    }
    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        if (maxValueMultiPlayer[i] != r*r*r) {
            printf("%s (%d):invalid max rate count: %d\n", __FUNCTION__, __LINE__, maxValueMultiPlayer[i]);
            return false;
        }
    }

    memset(maxValueMultiPlayer, 0, sizeof(maxValueMultiPlayer));
    r = COUNTER_PART_RATE_MULTIPLIER_STEP_COUNT;
    for (partIndex=0;partIndex<r*r*r*r;partIndex++) {
        memset(multiplayer, -1, sizeof(multiplayer));
        generateRatesCounterPartValue(partIndex, multiplayer);
        if (!isValidRates(multiplayer)) {
            printf("%s (%d):invalid rate: %llu\n", __FUNCTION__, __LINE__, partIndex);
            return false;
        }
        setMaxValue(multiplayer, maxValueMultiPlayer);
    }
    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        if (maxValueMultiPlayer[i] != r*r*r) {
            printf("%s (%d):invalid max rate count: %d\n", __FUNCTION__, __LINE__, maxValueMultiPlayer[i]);
            return false;
        }
    }
    return true;
}

#ifdef CUDA_COMPILE
__host__
#endif
bool UnitSearchPartIndex::isValidRates(const int *multiplayer)
{
    int i;
    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        if (multiplayer[i] <= 0 || multiplayer[i] > 100) {
            return false;
        }
    }
    return true;
}

#ifdef CUDA_COMPILE
__host__
#endif
void UnitSearchPartIndex::setMaxValue(const int *multiplayer, int *maxValueMultiPlayer)
{
    int i;
    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        if (multiplayer[i] == 100) {
            maxValueMultiPlayer[i]++;
        }
    }
}
