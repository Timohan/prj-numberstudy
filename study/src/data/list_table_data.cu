/**
 * @file list_table_data.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief Contains list table data functionality
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "list_table_data.h"
#include "../define_values.h"
#include "../study_cuda/generate_matrix_lines.h"

#ifdef CUDA_COMPILE
__global__
#endif
/**
 * @brief init list table data
 * 
 * @param list [in/out] init table data
 * @param tableDataCount table data count
 */
void initListTableData(ListTableData *list, unsigned int tableDataCount)
{
    list->m_listTableDataCount = tableDataCount;
    list->m_listTableData = new TableData[tableDataCount];
}

#ifdef CUDA_COMPILE
__global__
#endif
/**
 * @brief delete pointers of ListTableData *list
 * 
 * @param list [in/out] pointers are deleted from here
 */
void clearListTableData(ListTableData *list)
{
    if (list->m_listTableDataCount) {
        for (unsigned int i=0;i<list->m_listTableDataCount;i++) {
            clearTableData(&list->m_listTableData[list->m_listTableDataCount-i-1]);
        }
        delete[] list->m_listTableData;
        list->m_listTableData = nullptr;
    }
    list->m_listTableDataCount = 0;
}

#ifdef CUDA_COMPILE
__global__
#endif
/**
 * @brief
 * 
 * @param list [in/out] set table data base into list
 * @param data [in] table data
 * @param tableIndex [in] table index of table data
 * @param counterPartCount table count
 */
void setListTableDataTable(ListTableData *list, const TableData data, const unsigned tableIndex, unsigned int counterPartCount)
{
    setTableData(&data, &list->m_listTableData[tableIndex], counterPartCount);
}

#ifdef CUDA_COMPILE
__global__
#endif
/**
 * @brief fills value to cell
 * 
 * @param list [in/out] fills value to table cell
 * @param value value to cell
 * @param columnIndex column index of cell
 * @param rowIndex row index of cell
 * @param tableIndex table index of cell
 * @param cellIndex cell index of cell
 * @param counterPartCount table count
 */
void setListTableDataTableCell(ListTableData *list, double value, unsigned int columnIndex, unsigned int rowIndex, const unsigned int tableIndex, const unsigned int cellIndex, unsigned int counterPartCount)
{
    setTableDataTableCell(&list->m_listTableData[tableIndex], value, columnIndex, rowIndex, cellIndex, counterPartCount);
}

#ifdef CUDA_COMPILE
__global__
#endif
/**
 * @brief 
 * 
 * @param list [in/out] set counter part pointers to each cell for this list
 * @param destinationTableDataIndex destination table index to fill counter parts
 */
void setListTableDataCounterParts(ListTableData *list, const unsigned int destinationTableDataIndex)
{
    unsigned int destinationCellIndex;
    unsigned int sourceTableIndex;
    for (destinationCellIndex=0;destinationCellIndex<list->m_listTableData[destinationTableDataIndex].m_listTableCellCount;destinationCellIndex++) {
        for (sourceTableIndex=0;sourceTableIndex<list->m_listTableDataCount;sourceTableIndex++) {
            if (destinationTableDataIndex == sourceTableIndex) {
                continue;
            }
            setTableDataCounterPart(&list->m_listTableData[sourceTableIndex],
                sourceTableIndex,
                &list->m_listTableData[destinationTableDataIndex].m_listTableCell[destinationCellIndex]);
        }
    }
}

#ifdef CUDA_COMPILE
__global__
#endif
/**
 * @brief
 * 
 * @param list [in/out] set next and previous next cell points to this
 */
void setListTableDataCellPreviousNextCells(ListTableData *list)
{
    for (unsigned int i=0;i<list->m_listTableDataCount;i++) {
        setTableDataCellPreviousNextCells(&list->m_listTableData[i]);
    }
}

#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief init list table data
 * 
 * @param list [in/out] init table data
 * @param tableDataCount table data count
 */
void host_initListTableData(ListTableData *list, unsigned int tableDataCount)
{
    list->m_listTableDataCount = tableDataCount;
    list->m_listTableData = new TableData[tableDataCount];
}

#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief delete pointers of ListTableData *list
 * 
 * @param list [in/out] pointers are deleted from here
 */
void host_clearListTableData(ListTableData *list)
{
    if (list->m_listTableDataCount) {
        for (unsigned int i=0;i<list->m_listTableDataCount;i++) {
            clearTableDataCounterPart(&list->m_listTableData[list->m_listTableDataCount-i-1]);
        }
        for (unsigned int i=0;i<list->m_listTableDataCount;i++) {
            clearTableData(&list->m_listTableData[list->m_listTableDataCount-i-1]);
        }
        delete[] list->m_listTableData;
        list->m_listTableData = nullptr;
    }
    list->m_listTableDataCount = 0;
}

#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief
 * 
 * @param list [in/out] set table data base into list
 * @param data [in] table data
 * @param tableIndex [in] table index of table data
 * @param counterPartCount table count
 */
void host_setListTableDataTable(ListTableData *list, const TableData *data, const unsigned tableIndex, unsigned int counterPartCount)
{
    setTableData(data, &list->m_listTableData[tableIndex], counterPartCount);
}

#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief fills value to cell
 * 
 * @param list [in/out] fills value to table cell
 * @param value value to cell
 * @param columnIndex column index of cell
 * @param rowIndex row index of cell
 * @param tableIndex table index of cell
 * @param cellIndex cell index of cell
 * @param counterPartCount table count
 */
void host_setListTableDataTableCell(ListTableData *list, double value, unsigned int columnIndex, unsigned int rowIndex,
                                    const unsigned int tableIndex, const unsigned int cellIndex, unsigned int counterPartCount)
{
    setTableDataTableCell(&list->m_listTableData[tableIndex], value, columnIndex, rowIndex, cellIndex, counterPartCount);
}

#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief 
 * 
 * @param list [in/out] set counter part pointers to each cell for this list
 * @param destinationTableDataIndex destination table index to fill counter parts
 */
void host_setListTableDataCounterParts(ListTableData *list, const unsigned int destinationTableDataIndex)
{
    unsigned int destinationCellIndex;
    unsigned int sourceTableIndex;
    for (destinationCellIndex=0;destinationCellIndex<list->m_listTableData[destinationTableDataIndex].m_listTableCellCount;destinationCellIndex++) {
        for (sourceTableIndex=0;sourceTableIndex<list->m_listTableDataCount;sourceTableIndex++) {
            if (destinationTableDataIndex == sourceTableIndex) {
                continue;
            }
            setTableDataCounterPart(&list->m_listTableData[sourceTableIndex],
                sourceTableIndex,
                &list->m_listTableData[destinationTableDataIndex].m_listTableCell[destinationCellIndex]);
        }
    }
}

#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief
 * 
 * @param list [in/out] set next and previous next cell points to this
 */
void host_setListTableDataCellPreviousNextCells(ListTableData *list)
{
    for (unsigned int i=0;i<list->m_listTableDataCount;i++) {
        setTableDataCellPreviousNextCells(&list->m_listTableData[i]);
    }
}

#ifdef CUDA_COMPILE
__host__
#endif
/**
 * @brief
 * generates pre-calculated acceptable counter part for
 * each previous index, so it doesn't need to try study
 * this counter part and waste a time
 * 
 * @param list [in/out] accepted values are set to this
 * @param resultColumnIndex [in] primary result column
 */
void host_generateAcceptableTableData(ListTableData *list, const int resultColumnIndex)
{
    int previousIndex;
    unsigned int counterPartIndex;
    bool accepted;
    int typeIndex;
    for (previousIndex=MIN_PREVIOUS_NEXT_INDEX;previousIndex<=MAX_PREVIOUS_NEXT_INDEX;previousIndex++) {
        for (counterPartIndex=1;counterPartIndex<list->m_listTableDataCount;counterPartIndex++) {
            for (typeIndex=0;typeIndex<static_cast<int>(CounterPartMathType::CounterPartMathType_Count);typeIndex++) {
                accepted = isAcceptableGenerating(list,
                                counterPartIndex,
                                resultColumnIndex,
                                previousIndex,
                                static_cast<CounterPartMathType>(typeIndex));
                list->m_listTableData[counterPartIndex].m_previousNextIndex[previousIndex-MIN_PREVIOUS_NEXT_INDEX] = accepted;
                if (!accepted) {
                    break;
                }
            } 
        }
    }
}

#ifdef CUDA_COMPILE
__global__
#endif
/**
 * @brief generates pre-calculated acceptable counter part for
 * each previous index, so it doesn't need to try study
 * this counter part and waste a time
 * 
 * @param list [in/out] accepted values are set to this
 * @param resultColumnIndex [in] primary result column
 */
void generateAcceptableTableData(ListTableData *list, const int resultColumnIndex)
{
    int previousIndex;
    unsigned int counterPartIndex;
    bool accepted;
    int typeIndex;
    for (previousIndex=MIN_PREVIOUS_NEXT_INDEX;previousIndex<=MAX_PREVIOUS_NEXT_INDEX;previousIndex++) {
        for (counterPartIndex=1;counterPartIndex<list->m_listTableDataCount;counterPartIndex++) {
            for (typeIndex=0;typeIndex<static_cast<int>(CounterPartMathType::CounterPartMathType_Count);typeIndex++) {
                accepted = isAcceptableGenerating(list,
                                counterPartIndex,
                                resultColumnIndex,
                                previousIndex,
                                static_cast<CounterPartMathType>(typeIndex));
                list->m_listTableData[counterPartIndex].m_previousNextIndex[previousIndex-MIN_PREVIOUS_NEXT_INDEX] = accepted;
                if (!accepted) {
                    break;
                }
            }
        }
    }
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief get max row count from table datas
 * 
 * @param list max row count from list
 * @return unsigned int max row count
 */
unsigned int getMaxRowCountFromListTableData(const ListTableData *list)
{
    unsigned int ret = 1;
    unsigned int tmp;
    for (unsigned int i=0;i<list->m_listTableDataCount;i++) {
        tmp = getMaxRowCountFromTableData(&list->m_listTableData[i]);
        if (tmp > ret) {
            ret = tmp;
        }
    }
    return ret;
}
