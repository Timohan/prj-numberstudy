/**
 * @file best_result_storage.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief stores best results
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef BEST_RESULT_STORAGE_H_
#define BEST_RESULT_STORAGE_H_

#include "../define_values.h"
#include <string>

class BestResultStorage
{
public:
    BestResultStorage();
    void setBestResult(const int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                       const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT],
                       const uint64_cu partIndex, const double result, int resultColumnIndex,
                       const double *listCalculatedDotProduct /* [MAX_MATRIX_COLUMS] */ );
    void setBestResultPrimaryOnly(const double result, int resultColumnIndex,
                                  const double *listCalculatedDotProduct /* [MAX_MATRIX_COLUMS] */ );
    void setBestResultPrimaryFineTuneOnly(const uint64_cu partIndex,
                                          const double result, int resultColumnIndex,
                                          const double *listCalculatedDotProduct /* [MAX_MATRIX_COLUMS] */ );
    void setBestResultCounterPartFineTuneOnly(const unsigned int counterPartIndex,
                                                             const uint64_cu partIndex,
                                                             const double result, int resultColumnIndex,
                                                             const double *listCalculatedDotProduct /* [MAX_MATRIX_COLUMS] */ );
    void setBestResultNewCounterPart(const uint64_cu partIndex, const double result, int resultColumnIndex,
                                                    const double *listCalculatedDotProduct /* [MAX_MATRIX_COLUMS] */,
                                                    int newCounterPartPositionIndex,
                                                    int newCounterPartIndex,
                                                    int newPreviousIndex,
                                                    CounterPartMathType newCounterPartMathType);
    void save(const std::string &resultFile);
    void load(const std::string &resultFile);

    int *getListCounterPartIndex();
    int getListCounterPartCount();
    CounterPartMathType *getListCounterPartMathTypeIndex();
    int *getCounterPartMultiplierPrimary();
    int *getCounterPartMultiplierCounter();
    int *getPreviousIndex();
    int getResultColumnIndex();
    double *getCalculatedDotProductBestRateValues();
    double getCurrentBestResult();
    void print();

private:
    int m_listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT];
    uint64_cu m_partIndex = 0;
    double m_currentBestResult = DEFAULT_BEST_VALUE;
    int m_resultColumnIndex = 0;
    int m_bestCounterPartMultipliersPrimary[MAX_MATRIX_COLUMS];
    int m_bestCounterPartMultipliersCounter[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT];
    int m_bestPreviousIndex[MAX_COUNTER_PART_INDEX_COUNT];
    double m_listCalculatedDotProductBestRateValues[MAX_MATRIX_COLUMS];
    CounterPartMathType m_listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT];
};

#endif // DATA_COUNTER_PART_H_