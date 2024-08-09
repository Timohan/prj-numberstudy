/**
 * @file generate_matrix_lines.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief generate matrix line for study
 * 
 * @copyright Copyright (c) 2024
 */
#include "generate_matrix_lines.h"
#include <stdio.h>
#include "generate_matrix_lines_calculate.h"

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief get counter part for calculate matrix lines values
 * 
 * @param tableDataCell primary value cell, find counter part from this cell
 * @param counterPartIndex counter part index
 * @param previousIndex previous index
 * @return const TableDataCell* counter part cell, if nullptr -> then not found
 */
const TableDataCell *generateMatrixLinesGetCounterPart(const TableDataCell *tableDataCell,
                                                       const int counterPartIndex,
                                                       const int previousIndex)
{
    if (previousIndex == 0) {
        if (tableDataCell->m_listCounterPart[counterPartIndex].m_initCompleted) {
            return tableDataCell->m_listCounterPart[counterPartIndex].m_counterPart;
        }
        return nullptr;
    }

    if (tableDataCell->m_listCounterPart[counterPartIndex].m_initCompleted) {
        const TableDataCell *counterPartCell = getPreviousNextCell(tableDataCell->m_listCounterPart[counterPartIndex].m_counterPart, previousIndex);
        if (counterPartCell != nullptr) {
            return counterPartCell;
        }
    }
    const TableDataCell *ownPartCell = getPreviousNextCell(tableDataCell, previousIndex);
    if (ownPartCell == nullptr) {
        return nullptr;
    }

    if (ownPartCell->m_listCounterPart[counterPartIndex].m_initCompleted) {
        return ownPartCell->m_listCounterPart[counterPartIndex].m_counterPart;
    }

    return nullptr;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief generate matrix line column value from cell
 * 
 * @param bestCounterPartMultiplier counter part multipliers - [0] is primary multiplier
 * @param previousIndex list of previous index
 * @param listCounterPartIndex list of counter parts
 * @param matrixLines [out] matrix line value for column is set here
 * @param matrixLineColumnIndex matrix line column
 * @param tableDataCell primary cell
 * @param listCounterPartMathType math type for calculates
 * @return true if calculation was possible.
 * @return false if not
 */
bool generateMatrixLineValueFromCell(const int bestCounterPartMultiplier[MAX_COUNTER_PART_INDEX_COUNT+1],
                                     const int *previousIndex,
                                     const int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                                     CudaMatrix4xX::Matrix4xX *matrixLines,
                                     const unsigned int matrixLineColumnIndex,
                                     const TableDataCell *tableDataCell,
                                     const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT])
{
    unsigned int i;
    double value = tableDataCell->m_value*static_cast<double>(bestCounterPartMultiplier[0])/100.0;
    const TableDataCell *counterPartPrevious = nullptr;

    for (i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        if (listCounterPartIndex[i] <= 0) {
            continue;
        }
        const TableDataCell *counterPartCell = generateMatrixLinesGetCounterPart(tableDataCell,
                                                       listCounterPartIndex[i],
                                                       previousIndex[i]);
        if (!counterPartCell) {
            return false;
        }

        switch (listCounterPartMathType[i]) {
            case CounterPartMathType::CounterPartMathType_Plus_Normal:
            case CounterPartMathType::CounterPartMathType_Count:
            default:
                value += static_cast<double>(bestCounterPartMultiplier[i+1])*counterPartCell->m_value/100.0;
                break;
            case CounterPartMathType::CounterPartMathType_Plus_PreviousMinus:
                counterPartPrevious = getPreviousNextCell(counterPartCell, -1);
                if (!counterPartPrevious) {
                    return false;
                }
                value += static_cast<double>(bestCounterPartMultiplier[i+1])*(counterPartCell->m_value-counterPartPrevious->m_value)/100.0;
                break;
        }
    }

    matrixLines->m_listMatrixLines[ matrixLines->m_listMatrixLinesCount ].m[matrixLineColumnIndex] = value;
    return true;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief generates matrx line for study dot product values
 * 
 * @param bestCounterPartMultipliersPrimary primary value multipliers
 * @param bestCounterPartMultipliersCounter counter part multipliers
 * @param previousIndex list of previous indexes for counter pats
 * @param listCounterPartIndex list of counter parts
 * @param matrixLines [out] generated matrix lines
 * @param tableData table data of primary table
 * @param currentTableCellIndex cell index from tableData
 * @param listCounterPartMathType counter part math types
 * @return true if acceptable.
 * @return false if not
 */
bool generateMatrixLine(const int bestCounterPartMultipliersPrimary[MAX_MATRIX_COLUMS],
                        const int bestCounterPartMultipliersCounter[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT],
                        const int *previousIndex,
                        const int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                        CudaMatrix4xX::Matrix4xX *matrixLines,
                        const TableData *tableData,
                        const unsigned int currentTableCellIndex,
                        const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT])
{
    if (currentTableCellIndex >= tableData->m_listTableCellCount) {
        return false;
    }

    int bestCounterPartMultiplierWithCounter[MAX_COUNTER_PART_INDEX_COUNT+1];

    unsigned int cellIndex, cellIndexForMultiplier;
    const TableDataCell *cell = &tableData->m_listTableCell[currentTableCellIndex];
    matrixLines->m_listValue[ matrixLines->m_listMatrixLinesCount ] = cell->m_value;

    for (cellIndex=0;cellIndex<MAX_MATRIX_COLUMS;cellIndex++) {
        cell = getPreviousNextCell(cell, -1);
        if (!cell) {
            return false;
        }
        bestCounterPartMultiplierWithCounter[0] = bestCounterPartMultipliersPrimary[MAX_MATRIX_COLUMS-cellIndex-1];

        for (cellIndexForMultiplier=0;cellIndexForMultiplier<MAX_COUNTER_PART_INDEX_COUNT;cellIndexForMultiplier++) {
            bestCounterPartMultiplierWithCounter[cellIndexForMultiplier+1] = bestCounterPartMultipliersCounter[ (MAX_MATRIX_COLUMS-cellIndex-1) + cellIndexForMultiplier*MAX_MATRIX_COLUMS ];
        }

        if (!generateMatrixLineValueFromCell(bestCounterPartMultiplierWithCounter,
                                        previousIndex,
                                        listCounterPartIndex,
                                        matrixLines,
                                        MAX_MATRIX_COLUMS-cellIndex-1,
                                        cell,
                                        listCounterPartMathType)) {
            return false;
        }
    }
    matrixLines->m_listMatrixLinesCount++;
    return true;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief checks if searching is acceptable for individual
 * counter part to prevent too much calculation
 * 
 * @param list table data list
 * @param counterPartIndex counter part index to search
 * @param resultColumnIndex end result column index
 * @param previousIndex previous index for counter part cell search
 * @param listCounterPartMathType math type to calculate counter part
 * @return true if acceptable.
 * @return false if not
 */
bool isAcceptableGenerating(const ListTableData *list,
                            const int counterPartIndex,
                            const int resultColumnIndex,
                            const int previousIndex,
                            const CounterPartMathType listCounterPartMathType)
{
    int resultColumnIndexCalculate = resultColumnIndex - 1;
    if (resultColumnIndexCalculate < 0) {
        resultColumnIndexCalculate = findLargestColumnIndex(&list->m_listTableData[0]);
    }

    unsigned int ownIndex = getTableDataTableCellIndexByColumnIndexLast(&list->m_listTableData[0], resultColumnIndexCalculate);
    if (ownIndex == __UINT32_MAX__) {
        return false;
    }

    const TableDataCell *counterPartCell = generateMatrixLinesGetCounterPart(&list->m_listTableData[0].m_listTableCell[ownIndex],
                                                       counterPartIndex,
                                                       previousIndex);
    if (!counterPartCell) {
        return false;
    }
    switch (listCounterPartMathType) {
        case CounterPartMathType::CounterPartMathType_Count:
        case CounterPartMathType::CounterPartMathType_Plus_Normal:
            break;
        case CounterPartMathType::CounterPartMathType_Plus_PreviousMinus:
            if (!getPreviousNextCell(counterPartCell, -1)) {
                return false;
            }
            break;
    }
    return counterPartCell->m_rowIndex >= 
        list->m_listTableData[0].m_listTableCell[ownIndex].m_rowIndex;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief checks if it's acceptable for generating
 * for all previous indexes, max types and so on.
 * 
 * @param list table data list
 * @param listCounterPartIndex list of counter part indexes
 * @param resultColumnIndex column index of result to study
 * @param previousIndex list of previous index's for each counter part
 * @param listCounterPartMathType list of math types for each counter part
 * @return true if acceptable.
 * @return false if not
 */
bool isAcceptableGeneratings(const ListTableData *list,
                             const int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                             const int resultColumnIndex,
                             const int previousIndex[MAX_COUNTER_PART_INDEX_COUNT],
                             const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT])
{
    int i;
    for (i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        if (listCounterPartIndex[i] <= 0) {
            continue;
        }
        if (!isAcceptableGenerating(list,
                            listCounterPartIndex[i],
                            resultColumnIndex,
                            previousIndex[i],
                            listCounterPartMathType[i])) {
            return false;
        }
    }
    return true;
}


#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief generates matrix lines for study
 * 
 * @param list table data list
 * @param bestCounterPartMultipliersPrimary current best multipliers for primary value
 * @param bestCounterPartMultipliersCounter current best multipliers for counter part values
 * @param previousIndex list of previous indexes for counter parts
 * @param resultColumnIndex result column index
 * @param listCounterPartIndex list of counter part index
 * @param matrixLines [in/out] matrix lines to fill
 * @param listCounterPartMathType counter part math types
 * @return true if matrix lines were added.
 * @return false if not
 */
bool generateMatrixLines(const ListTableData *list,
                         const int bestCounterPartMultipliersPrimary[MAX_MATRIX_COLUMS],
                         const int bestCounterPartMultipliersCounter[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT],
                         const int previousIndex[MAX_MATRIX_COLUMS],
                         const int resultColumnIndex,
                         const int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                         CudaMatrix4xX::Matrix4xX *matrixLines,
                         const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT])
{
    unsigned int currentTableCellIndex = 0;
    currentTableCellIndex = getTableDataTableCellIndexByColumnIndex(&list->m_listTableData[0], 0, resultColumnIndex);
    if (__UINT32_MAX__ == currentTableCellIndex) {
        return false;
    }
    if (!isAcceptableGeneratings(list,
                             listCounterPartIndex,
                             resultColumnIndex,
                             previousIndex,
                             listCounterPartMathType)) {
        return false;
    }

    bool ret = false;
    while (1) {
        ret = generateMatrixLine(bestCounterPartMultipliersPrimary,
                        bestCounterPartMultipliersCounter,
                        previousIndex,
                        listCounterPartIndex,
                        matrixLines,
                        &list->m_listTableData[0],
                        currentTableCellIndex,
                        listCounterPartMathType);

        currentTableCellIndex = getTableDataTableCellIndexByColumnIndex(&list->m_listTableData[0], currentTableCellIndex+1, resultColumnIndex);
        if (__UINT32_MAX__ == currentTableCellIndex) {
            break;
        }
    }

    return ret && matrixLines->m_listMatrixLinesCount > MIN_ROW_LINE_COUNT_FOR_MAKE_MATRIX_CALC;
}