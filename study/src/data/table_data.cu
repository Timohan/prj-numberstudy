/**
 * @file table_data.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief handles table data struct
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "table_data.h"
#include <stdio.h>

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief set table data to destionation
 * 
 * @param source this table data to destination
 * @param destination cell value will be filled to tableData
 * @param tableDataCount table data count
 */
void setTableData(const TableData *source, TableData *destination, unsigned int tableDataCount)
{
    destination->m_listTableCellCount = source->m_listTableCellCount;
    destination->m_listTableCell = new TableDataCell[source->m_listTableCellCount];
    for (unsigned int i=0;i<source->m_listTableCellCount;i++) {
        destination->m_listTableCell[i].m_listCounterPartCount = tableDataCount;
    }
    for (unsigned int i=0;i<MAX_PREVIOUS_NEXT_INDEX-MIN_PREVIOUS_NEXT_INDEX+1;i++) {
        destination->m_previousNextIndex[i] = true;
    }
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief fill values to cell 
 * 
 * @param destination cell value will be filled to tableData
 * @param value value of cell
 * @param columnIndex column index of cell
 * @param rowIndex row index of cell
 * @param cellIndex cell index
 * @param tableDataCount table data count
 */
void setTableDataTableCell(TableData *destination, double value, unsigned int columnIndex, unsigned int rowIndex, const unsigned int cellIndex, const unsigned int tableDataCount)
{
    setTableDataCell(value, columnIndex, rowIndex, &destination->m_listTableCell[cellIndex], tableDataCount);
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief fills counter part pointer to destionation cell
 * from table data
 * 
 * @param source counter part cell from this table data
 * @param sourceDataTableIndex table data index of source
 * @param destination counter part will be filled to this cell
 */
void setTableDataCounterPart(const TableData *source, const unsigned int sourceDataTableIndex, TableDataCell *destination)
{
    for (unsigned int i=0;i<source->m_listTableCellCount;i++) {
        if (destination->m_columnIndex == source->m_listTableCell[i].m_columnIndex
            && destination->m_rowIndex == source->m_listTableCell[i].m_rowIndex) {
            setTableDataCellCounterPart(destination, sourceDataTableIndex, &source->m_listTableCell[i]);
            return;
        }
    }
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief pre-sets table data cells previous and next cells
 * 
 * @param tableData [in/out] previous and next cells filled to this table data
 */
void setTableDataCellPreviousNextCells(const TableData *tableData)
{
    if (tableData->m_listTableCellCount <= 1) {
        return;
    }

    for (unsigned int i=0;i<tableData->m_listTableCellCount;i++) {
        if (i == 0) {
            tableData->m_listTableCell[i].m_nextTableDataCell.m_initCompleted = true;
            tableData->m_listTableCell[i].m_nextTableDataCell.m_counterPart = &tableData->m_listTableCell[i+1];
        } else if (i+1 == tableData->m_listTableCellCount) {
            tableData->m_listTableCell[i].m_previousTableDataCell.m_initCompleted = true;
            tableData->m_listTableCell[i].m_previousTableDataCell.m_counterPart = &tableData->m_listTableCell[i-1];
        } else {
            tableData->m_listTableCell[i].m_previousTableDataCell.m_initCompleted = true;
            tableData->m_listTableCell[i].m_nextTableDataCell.m_initCompleted = true;
            tableData->m_listTableCell[i].m_previousTableDataCell.m_counterPart = &tableData->m_listTableCell[i-1];
            tableData->m_listTableCell[i].m_nextTableDataCell.m_counterPart = &tableData->m_listTableCell[i+1];
        }
    }
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief clear table data
 * 
 * @param tableData [in/out] clears this table data
 */
void clearTableData(TableData *tableData)
{
    if (tableData->m_listTableCell) {
        for (unsigned int i=0;i<tableData->m_listTableCellCount;i++) {
            clearTableDataCell(&tableData->m_listTableCell[i]);
        }
        delete[] tableData->m_listTableCell;
        tableData->m_listTableCell = nullptr;
    }
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief get cell index
 * 
 * @param tableData index of cell from this table data
 * @param fromIndex from index (starts searching from this index)
 * @param rowIndex row index of cell
 * @param columnIndex column index of cell
 * @return unsigned int index of last cell
 * if not found then __UINT32_MAX__
 */
unsigned int getTableDataTableCellIndex(const TableData *tableData, const unsigned int fromIndex, const unsigned int rowIndex, const unsigned int columnIndex)
{
    for (unsigned int i=fromIndex;i<tableData->m_listTableCellCount;i++) {
        if (tableData->m_listTableCell[i].m_rowIndex == rowIndex
            && tableData->m_listTableCell[i].m_columnIndex == columnIndex) {
            return i;
        }
    }
    return __UINT32_MAX__;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief get first cell index by column index from fromIndex
 * 
 * @param tableData index of cell from this table data
 * @param fromIndex from index (starts searching from this index)
 * @param columnIndex column index to search
 * @return unsigned int index of last cell
 * if not found then __UINT32_MAX__
 */
unsigned int getTableDataTableCellIndexByColumnIndex(const TableData *tableData, const unsigned int fromIndex, const unsigned int columnIndex)
{
    for (unsigned int i=fromIndex;i<tableData->m_listTableCellCount;i++) {
        if (tableData->m_listTableCell[i].m_columnIndex == columnIndex) {
            return i;
        }
    }
    return __UINT32_MAX__;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief get last index of cell by column index
 * 
 * @param tableData last index of cell from this table data
 * @param columnIndex column index to search
 * @return unsigned int index of last cell
 * if not found then __UINT32_MAX__
 */
unsigned int getTableDataTableCellIndexByColumnIndexLast(const TableData *tableData, const unsigned int columnIndex)
{
    unsigned int ret = __UINT32_MAX__;
    for (unsigned int i=0;i<tableData->m_listTableCellCount;i++) {
        if (tableData->m_listTableCell[i].m_columnIndex == columnIndex) {
            ret = i;
        }
    }
    return ret;
}

/**
 * @brief clear table data pointers
 * 
 * @param tableData [in/out] pointers are deleted from this table data
 */
void host_clearTableData(TableData *tableData)
{
    if (tableData->m_listTableCell) {
        for (unsigned int i=0;i<tableData->m_listTableCellCount;i++) {
            clearTableDataCell(&tableData->m_listTableCell[i]);
        }
        delete[] tableData->m_listTableCell;
        tableData->m_listTableCell = nullptr;
    }
}


#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief find largest column index for table data
 * 
 * @param tableData largest column index from this table data
 * @return unsigned int largest column index
 */
unsigned int findLargestColumnIndex(const TableData *tableData)
{
    unsigned int ret = 0;
    for (unsigned int i=0;i<tableData->m_listTableCellCount;i++) {
        if (tableData->m_listTableCell[i].m_columnIndex > ret) {
            ret = tableData->m_listTableCell[i].m_columnIndex;
        }
    }
    return ret;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief get if valid previous index to avoid extra calculation
 * 
 * @param source previous index of this table data
 * @param previousIndex previous index
 * @return true if it's acceptable (valid) previous index for study.
 * @return false if it's not
 */
bool isValidPreviousNextIndex(const TableData *source, const int previousIndex)
{
    if (previousIndex == __INT32_MAX__) {
        for (int i=0;i<=MAX_PREVIOUS_NEXT_INDEX-MIN_PREVIOUS_NEXT_INDEX;i++) {
            if (source->m_previousNextIndex[i]) {
                return true;
            }
        }
        return false;
    }
    return source->m_previousNextIndex[previousIndex-MIN_PREVIOUS_NEXT_INDEX];
}


#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief clears table data counter part pointers
 * 
 * @param tableData [in/out] clears table data pointers
 */
void clearTableDataCounterPart(TableData *tableData)
{
    if (tableData->m_listTableCell) {
        for (unsigned int i=0;i<tableData->m_listTableCellCount;i++) {
            clearTableDataCellCounterPart(&tableData->m_listTableCell[i]);
        }
    }
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief get max row count from table data
 * 
 * @param list max row count from list
 * @return unsigned int max row count
 */
unsigned int getMaxRowCountFromTableData(const TableData *tableData)
{
    if (tableData->m_listTableCellCount == 0) {
        return 0;
    }
    return tableData->m_listTableCell[tableData->m_listTableCellCount-1].m_rowIndex+1;
}
