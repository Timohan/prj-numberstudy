/**
 * @file generate_matrix_lines_calculate.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief generates matrix lines for calculate
 * 
 * @copyright Copyright (c) 2024
 */
#include "generate_matrix_lines_calculate.h"
#include <stdio.h>
#include "generate_matrix_lines.h"

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief generate matrix line for calculate
 * 
 * @param bestCounterPartMultipliersPrimary current best multipliers for primary value
 * @param bestCounterPartMultipliersCounter current best multipliers for counter part values
 * @param previousIndex list of previous indexes for counter parts
 * @param listCounterPartIndex column index for result
 * @param matrixLines [in/out] matrix lines to fill
 * @param tableData pointer to primary table data
 * @param currentTableCellIndex cell index
 * @param listCounterPartMathType counter part math types
 * @return true if matrix lines were added.
 * @return false if not
 */
bool generateMatrixLineCalculate(const int bestCounterPartMultipliersPrimary[MAX_MATRIX_COLUMS],
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

    unsigned int cellIndex, cellIndexForMultiplier;
    const TableDataCell *cell = &tableData->m_listTableCell[currentTableCellIndex];
    const TableDataCell *cellForward = getPreviousNextCell(cell, 1);
    if (cellForward) {
        matrixLines->m_listValue[ matrixLines->m_listMatrixLinesCount ] = cellForward->m_value;
    } else {
        matrixLines->m_listValue[ matrixLines->m_listMatrixLinesCount ] = -10000000.0;
    }

    int bestCounterPartMultiplierWithCounter[MAX_COUNTER_PART_INDEX_COUNT+1];

    for (cellIndex=0;cellIndex<MAX_MATRIX_COLUMS;cellIndex++) {
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
        cell = getPreviousNextCell(cell, -1);
        if (!cell) {
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
 * @brief generates matrix lines for calculate
 * 
 * @param list list of table data
 * @param bestCounterPartMultipliersPrimary current best multipliers for primary value
 * @param bestCounterPartMultipliersCounter current best multipliers for counter part values
 * @param previousIndex list of previous indexes for counter parts
 * @param resultColumnIndex column index for result
 * @param listCounterPartIndex list of counter parts (index)
 * @param matrixLines [in/out] matrix lines to fill
 * @param listCounterPartMathType counter part math types
 * @return true if matrix lines were added.
 * @return false if not
 */
bool generateMatrixLinesCalculate(const ListTableData *list,
                         const int bestCounterPartMultipliersPrimary[MAX_MATRIX_COLUMS],
                         const int bestCounterPartMultipliersCounter[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT],
                         const int *previousIndex,
                         const int resultColumnIndex,
                         const int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                         CudaMatrix4xX::Matrix4xX *matrixLines,
                         const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT])
{
    unsigned int currentTableCellIndex = 0;
    int currentLastColumnIndex = resultColumnIndex - 1;
    if (currentLastColumnIndex < 0) {
        currentLastColumnIndex = findLargestColumnIndex(&list->m_listTableData[0]);
    }

    currentTableCellIndex = getTableDataTableCellIndexByColumnIndex(&list->m_listTableData[0], 0, currentLastColumnIndex);
    if (__UINT32_MAX__ == currentTableCellIndex) {
        return false;
    }
    bool ret = false;

    while (1) {
        ret = generateMatrixLineCalculate(bestCounterPartMultipliersPrimary,
                        bestCounterPartMultipliersCounter,
                        previousIndex,
                        listCounterPartIndex,
                        matrixLines,
                        &list->m_listTableData[0],
                        currentTableCellIndex,
                        listCounterPartMathType);

        currentTableCellIndex = getTableDataTableCellIndexByColumnIndex(&list->m_listTableData[0], currentTableCellIndex+1, currentLastColumnIndex);
        if (__UINT32_MAX__ == currentTableCellIndex) {
            break;
        }
   }

    return ret && matrixLines->m_listMatrixLinesCount > MIN_ROW_LINE_COUNT_FOR_MAKE_MATRIX_CALC;
}