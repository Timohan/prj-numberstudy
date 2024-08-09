/**
 * @file calculate_results.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief calculates results
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "calculate_results.h"
#include "../study_cuda/generate_matrix_lines.h"
#include "../study_cuda/calculate_generated_matrix_lines.h"
#include "../study_cuda/generate_matrix_lines_calculate.h"

/**
 * @brief calculate result
 * 
 * @param dataTableLoader ref pointer to loader
 * @param bestResult ref pointer to result storage
 * @param resultFile saved result file
 */
void calculateResults(DataTableLoader &dataTableLoader,
    BestResultStorage &bestResult,
    const std::string &resultFile)
{
    uint64_t i;
    unsigned int cellIndex;
    bestResult.load(resultFile);
    ListTableData *d_listTableData = new ListTableData;
    host_initListTableData(d_listTableData, dataTableLoader.getListTableData()->m_listTableDataCount);

    for (i=0;i<dataTableLoader.getListTableData()->m_listTableDataCount;i++) {
        host_setListTableDataTable(d_listTableData, &dataTableLoader.getListTableData()->m_listTableData[i], i, dataTableLoader.getListTableData()->m_listTableDataCount);
        for (cellIndex=0;cellIndex<dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCellCount;cellIndex++) {
            host_setListTableDataTableCell(d_listTableData,
                    dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_value,
                    dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_columnIndex,
                    dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_rowIndex, i, 
                    cellIndex, dataTableLoader.getListTableData()->m_listTableDataCount);
        }
    }

    for (i=0;i<dataTableLoader.getListTableData()->m_listTableDataCount;i++) {
        host_setListTableDataCounterParts(d_listTableData,  i);
    }

    host_setListTableDataCellPreviousNextCells(d_listTableData);

    unsigned int rowCount = getMaxRowCountFromListTableData(d_listTableData);
    CudaMatrix4xX::Matrix4xX matrixLines;
    CudaMatrix4xX::init(&matrixLines, rowCount);

    matrixLines.m_listMatrixLinesCount = 0;
    if (!generateMatrixLinesCalculate(d_listTableData,
                         bestResult.getCounterPartMultiplierPrimary(),
                         bestResult.getCounterPartMultiplierCounter(),
                         bestResult.getPreviousIndex(),
                         bestResult.getResultColumnIndex(),
                         bestResult.getListCounterPartIndex(),
                         &matrixLines,
                         bestResult.getListCounterPartMathTypeIndex())) {
        printf("%s (%d): Generating matrix lines failed\n", __FUNCTION__, __LINE__);
        return;
    }

    int *listCounterPart = bestResult.getListCounterPartIndex();
    printf("Primary value type for calculate %s\n", dataTableLoader.getTableType(0).c_str());

    for (i=0;i<MAX_COUNTER_PART_INDEX_COUNT;i++) {
        if (listCounterPart[i] == 0) {
            continue;
        }
        printf("Counter part type for calculate: %s\n", dataTableLoader.getTableType(listCounterPart[i]).c_str() );
    }

    CudaMatrix4x4::Matrix4x1 line;
    for (i=0;i<MAX_MATRIX_COLUMS;i++) {
        double *rates = bestResult.getCalculatedDotProductBestRateValues();
        line.m[i] = rates[i];
    }

    for (i=0;i<matrixLines.m_listMatrixLinesCount;i++) {
        double result = calculateMatrix4xXLine(&line,
                              &matrixLines,
                              i);
        if (std::abs(matrixLines.m_listValue[i] - DEFAULT_BEST_VALUE) <= 0
            || std::abs(matrixLines.m_listValue[i] + DEFAULT_BEST_VALUE) <= 0) {
            printf("Result for this row: %f (Original result: NEW)\n", result);
        } else {
            printf("Result for this row: %f (Original result: %f)\n", result, matrixLines.m_listValue[i]);
        }
    }

    host_clearListTableData(d_listTableData);
    delete d_listTableData;
    CudaMatrix4xX::clear(&matrixLines);
}