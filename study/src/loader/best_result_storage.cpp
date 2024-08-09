/**
 * @file best_result_storage.cpp
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief stores best results
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "best_result_storage.h"
#include <stdio.h>
#include <string.h>
#include <fstream>
#include "../search_part_index/search_part_index.h"

BestResultStorage::BestResultStorage()
{
    memset(m_listCounterPartIndex, 0, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
}

/**
 * @brief set best result
 * 
 * @param listCounterPartIndex list best counter part indexes
 * @param listCounterPartMathType list best part math types
 * @param partIndex current best part index
 * @param result current best result
 * @param resultColumnIndex column index of result
 * @param listCalculatedDotProduct list of dot products
 */
void BestResultStorage::setBestResult(const int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                                      const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT],
                                      const uint64_cu partIndex, const double result, int resultColumnIndex,
                                      const double *listCalculatedDotProduct /* [MAX_MATRIX_COLUMS] */ )
{
    if (listCounterPartIndex) {
        memcpy(m_listCounterPartIndex, listCounterPartIndex, sizeof(int)*MAX_COUNTER_PART_INDEX_COUNT);
    }
    m_partIndex = partIndex;
    m_currentBestResult = result;
    m_resultColumnIndex = resultColumnIndex;

    char currentBestText[256];
    char tmp[128];
    currentBestText[0] = '\0';
    for (int i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        sprintf(tmp, " %d", m_listCounterPartIndex[i]);
        strcat(currentBestText, tmp);
    }
    sprintf(tmp, " partIndex: %lld rates: ", partIndex);
    strcat(currentBestText, tmp);

    for (int i=0;i<MAX_MATRIX_COLUMS;i++) {
        sprintf(tmp, " %d", m_bestCounterPartMultipliersPrimary[i]);
        strcat(currentBestText, tmp);
    }
    for (int i=0;i<MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT;i++) {
        sprintf(tmp, " %d", m_bestCounterPartMultipliersCounter[i]);
        strcat(currentBestText, tmp);
    }

    strcat(currentBestText, "Math type:");
    for (int i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        m_listCounterPartMathType[i] = listCounterPartMathType[i];
        sprintf(tmp, " %d", static_cast<int>(listCounterPartMathType[i]));
        strcat(currentBestText, tmp);
    }

    strcat(currentBestText, "Prev:");
    for (int i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        sprintf(tmp, "  %d", m_bestPreviousIndex[i]);
        strcat(currentBestText, tmp);
    }

    strcat(currentBestText, "calculated dot product values:");
    for (int i=0;i<MAX_MATRIX_COLUMS;i++) {
        m_listCalculatedDotProductBestRateValues[i] = listCalculatedDotProduct[i];
        sprintf(tmp, " %f", m_listCalculatedDotProductBestRateValues[i]);
        strcat(currentBestText, tmp);
    }

    sprintf(tmp, " value: %f", result);
    strcat(currentBestText, tmp);
    printf("CurrentBest: %s\n", currentBestText);
}

/**
 * @brief set best result (primary value only) 
 * @param result current best result
 * @param resultColumnIndex column index of result
 * @param listCalculatedDotProduct list of dot products
 */
void BestResultStorage::setBestResultPrimaryOnly(const double result, int resultColumnIndex,
                                                 const double *listCalculatedDotProduct /* [MAX_MATRIX_COLUMS] */ )
{
    m_partIndex = 0;
    m_currentBestResult = result;
    m_resultColumnIndex = resultColumnIndex;

    for (int i=0;i<MAX_MATRIX_COLUMS;i++) {
        m_bestCounterPartMultipliersPrimary[i] = DEFAULT_PRIMARY_VALUE_MULTIPLIER;
    }

    for (int i=0;i<MAX_MATRIX_COLUMS;i++) {
        m_listCalculatedDotProductBestRateValues[i] = listCalculatedDotProduct[i];
    }
}

/**
 * @brief set best result new counter part
 * 
 * @param partIndex part index
 * @param result current best result
 * @param resultColumnIndex column index of result
 * @param listCalculatedDotProduct list of dot products
 * @param newCounterPartPositionIndex new counter part position index
 * @param newCounterPartIndex new counter part index
 * @param newPreviousIndex new previous part index
 * @param newCounterPartMathType new math type for counter part
 */
void BestResultStorage::setBestResultNewCounterPart(const uint64_cu partIndex, const double result, int resultColumnIndex,
                                                    const double *listCalculatedDotProduct /* [MAX_MATRIX_COLUMS] */,
                                                    int newCounterPartPositionIndex,
                                                    int newCounterPartIndex,
                                                    int newPreviousIndex,
                                                    CounterPartMathType newCounterPartMathType)
{
    int bestCounterPartMultipliers[MAX_MATRIX_COLUMS];

    m_partIndex = partIndex;
    m_currentBestResult = result;
    m_resultColumnIndex = resultColumnIndex;

    generateRatesCounterPartValue(partIndex, bestCounterPartMultipliers);

    m_listCounterPartIndex[newCounterPartPositionIndex] = newCounterPartIndex;
    m_bestPreviousIndex[newCounterPartPositionIndex] = newPreviousIndex;
    m_listCounterPartMathType[newCounterPartPositionIndex] = newCounterPartMathType;

    for (int i=0;i<MAX_MATRIX_COLUMS;i++) {
        m_listCalculatedDotProductBestRateValues[i] = listCalculatedDotProduct[i];
        m_bestCounterPartMultipliersCounter[MAX_MATRIX_COLUMS*newCounterPartPositionIndex+i] = bestCounterPartMultipliers[i];
    }
}

/**
 * @brief set best result for primary finetune
 * 
 * @param partIndex part index
 * @param result current best result
 * @param resultColumnIndex column index of result
 * @param listCalculatedDotProduct list of dot products
 */
void BestResultStorage::setBestResultPrimaryFineTuneOnly(const uint64_cu partIndex,
                                                         const double result, int resultColumnIndex,
                                                         const double *listCalculatedDotProduct /* [MAX_MATRIX_COLUMS] */ )
{
    m_partIndex = 0;
    m_currentBestResult = result;
    m_resultColumnIndex = resultColumnIndex;

    generateRatesPrimaryFineTuneValue(partIndex, m_bestCounterPartMultipliersPrimary);

    for (int i=0;i<MAX_MATRIX_COLUMS;i++) {
        m_listCalculatedDotProductBestRateValues[i] = listCalculatedDotProduct[i];
    }
}

/**
 * @brief set best result for counter part finetune values
 * 
 * @param counterPartIndex counter part index of new values
 * @param partIndex part index
 * @param result current best result
 * @param resultColumnIndex column index of result
 * @param listCalculatedDotProduct list of dot products
 */
void BestResultStorage::setBestResultCounterPartFineTuneOnly(const unsigned int counterPartIndex,
                                                             const uint64_cu partIndex,
                                                             const double result, int resultColumnIndex,
                                                             const double *listCalculatedDotProduct /* [MAX_MATRIX_COLUMS] */ )
{
    m_partIndex = 0;
    m_currentBestResult = result;
    m_resultColumnIndex = resultColumnIndex;

    int multiplier[MAX_MATRIX_COLUMS];

    generateRatesPrimaryFineTuneValue(partIndex, multiplier);
    for (int i=0;i<MAX_MATRIX_COLUMS;i++) {
        m_bestCounterPartMultipliersCounter[MAX_MATRIX_COLUMS*counterPartIndex+i] = multiplier[i];
    }

    for (int i=0;i<MAX_MATRIX_COLUMS;i++) {
        m_listCalculatedDotProductBestRateValues[i] = listCalculatedDotProduct[i];
    }
}


/**
 * @brief save results to file
 * 
 * @param resultFile result file
 */
void BestResultStorage::save(const std::string &resultFile)
{
    int i, tmp;
    std::ofstream writeFile(resultFile, std::ios::out | std::ios::binary);

    for (i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        writeFile.write(reinterpret_cast<char *>(&m_listCounterPartIndex[i]), sizeof(int));
        tmp = static_cast<int>(m_listCounterPartMathType[i]);
        writeFile.write(reinterpret_cast<char *>(&tmp), sizeof(int));
    }
    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        writeFile.write(reinterpret_cast<char *>(&m_listCalculatedDotProductBestRateValues[i]), sizeof(double));
    }
    for (int i=0;i<MAX_MATRIX_COLUMS;i++) {
        writeFile.write(reinterpret_cast<char *>(&m_bestCounterPartMultipliersPrimary[i]), sizeof(int));
    }
    for (int i=0;i<MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT;i++) {
        writeFile.write(reinterpret_cast<char *>(&m_bestCounterPartMultipliersCounter[i]), sizeof(int));
    }
    for (int i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        writeFile.write(reinterpret_cast<char *>(&m_bestPreviousIndex[i]), sizeof(int));
    }
    writeFile.write(reinterpret_cast<char *>(&m_currentBestResult), sizeof(double));
    writeFile.write(reinterpret_cast<char *>(&m_resultColumnIndex), sizeof(int));

    writeFile.close();

    print();
}

/**
 * @brief load results from file
 * 
 * @param resultFile result file
 */
void BestResultStorage::load(const std::string &resultFile)
{
    int i, tmp;
    std::ifstream infile;
    infile.open(resultFile);

    for (i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        infile.read(reinterpret_cast<char *>(&m_listCounterPartIndex[i]), sizeof(int));
        infile.read(reinterpret_cast<char *>(&tmp), sizeof(int));
        m_listCounterPartMathType[i] = static_cast<CounterPartMathType>(tmp);
    }
    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        infile.read(reinterpret_cast<char *>(&m_listCalculatedDotProductBestRateValues[i]), sizeof(double));
    }
    for (int i=0;i<MAX_MATRIX_COLUMS;i++) {
        infile.read(reinterpret_cast<char *>(&m_bestCounterPartMultipliersPrimary[i]), sizeof(int));
    }
    for (int i=0;i<MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT;i++) {
        infile.read(reinterpret_cast<char *>(&m_bestCounterPartMultipliersCounter[i]), sizeof(int));
    }
    for (int i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        infile.read(reinterpret_cast<char *>(&m_bestPreviousIndex[i]), sizeof(int));
    }
    infile.read(reinterpret_cast<char *>(&m_currentBestResult), sizeof(double));
    infile.read(reinterpret_cast<char *>(&m_resultColumnIndex), sizeof(int));

    infile.close();

    print();
}

/**
 * @brief 
 * 
 * @return int* list of counter parts
 */
int *BestResultStorage::getListCounterPartIndex()
{
    return m_listCounterPartIndex;
}

/**
 * @brief 
 * 
 * @return int* list of counter parts
 */
int BestResultStorage::getListCounterPartCount()
{
    int i;
    int ret = 0;
    for (i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        if (m_listCounterPartIndex[i] != 0) {
            ret++;
        }
    }
    return ret;
}


/**
 * @brief 
 * 
 * @return int* list of previous index values
 */
int *BestResultStorage::getPreviousIndex()
{
    return m_bestPreviousIndex;
}

/**
 * @brief 
 * 
 * @return double* list dot products (best value)
 */
double *BestResultStorage::getCalculatedDotProductBestRateValues()
{
    return m_listCalculatedDotProductBestRateValues;
}

/**
 * @brief 
 * 
 * @return int* counter part multipliers for primary values
 */
int *BestResultStorage::getCounterPartMultiplierPrimary()
{
    return m_bestCounterPartMultipliersPrimary;
}

/**
 * @brief 
 * 
 * @return int* counter part multipliers for counter part values
 */
int *BestResultStorage::getCounterPartMultiplierCounter()
{
    return m_bestCounterPartMultipliersCounter;
}

/**
 * @brief 
 * 
 * @return int result column index
 */
int BestResultStorage::getResultColumnIndex()
{
    return m_resultColumnIndex;
}

/**
 * @brief 
 * 
 * @return double current best result value
 */
double BestResultStorage::getCurrentBestResult()
{
    return m_currentBestResult;
}

/**
 * @brief 
 * 
 * @return CounterPartMathType* list of counter part math types
 */
CounterPartMathType *BestResultStorage::getListCounterPartMathTypeIndex()
{
    return m_listCounterPartMathType;
}

/**
 * @brief print values
 * 
 */
void BestResultStorage::print()
{
    int i, i2;
    char tmp[128];
    char currentText[1024+MAX_COUNTER_PART_INDEX_COUNT*128];
    currentText[0] = '\0';

    sprintf(tmp, "Study completed for column: %d", m_resultColumnIndex);
    strcat(currentText, tmp);


    sprintf(tmp, " PrimaryMultplier: ");
    strcat(currentText, tmp);
    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        sprintf(tmp, " %d", m_bestCounterPartMultipliersPrimary[i]);
        strcat(currentText, tmp);
    }

    strcat(currentText, "\nCounterPartIndex:\n");
    for (i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        if (m_listCounterPartIndex[i] == 0) {
            continue;
        }
        sprintf(tmp, " %d ", m_listCounterPartIndex[i]);
        strcat(currentText, tmp);
        for (i2=0;i2<MAX_MATRIX_COLUMS;i2++) {
            sprintf(tmp, "%d", m_bestCounterPartMultipliersCounter[i*MAX_MATRIX_COLUMS+i2]);
            strcat(currentText, tmp);
            if (i2 + 1 != MAX_MATRIX_COLUMS) {
                strcat(currentText, "/");
            }
        }

        sprintf(tmp, " Prev: %d ", m_bestPreviousIndex[i]);
        strcat(currentText, tmp);
        sprintf(tmp, "Type: %d\n", m_listCounterPartMathType[i]);
        strcat(currentText, tmp);
    }

    strcat(currentText, " DotProduct:");
    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        sprintf(tmp, " %f", m_listCalculatedDotProductBestRateValues[i]);
        strcat(currentText, tmp);
    }
    printf("%s\n", currentText);
}
