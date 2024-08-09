#ifndef UNITTEST_SEARCH_PART_INDEX_H
#define UNITTEST_SEARCH_PART_INDEX_H

class UnitSearchPartIndex
{
public:
#ifdef CUDA_COMPILE
__host__
#endif
    static bool search();

private:
#ifdef CUDA_COMPILE
__host__
#endif
    static bool isValidRates(const int *multiplayer);
    static void setMaxValue(const int *multiplayer, int *maxValueMultiPlayer);
};

#endif // UNITTEST_SEARCH_PART_INDEX_H
