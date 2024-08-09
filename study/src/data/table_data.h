/**
 * @file table_data.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief handles table data struct
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef TABLE_DATA_H
#define TABLE_DATA_H

#include "table_data_cell.h"
#include "../define_values.h"

struct TableData
{
    TableDataCell *m_listTableCell = nullptr;
    unsigned int m_listTableCellCount = 0;

    bool m_previousNextIndex[MAX_PREVIOUS_NEXT_INDEX-MIN_PREVIOUS_NEXT_INDEX+1];
};

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void setTableData(const TableData *source, TableData *destination, unsigned int tableDataCount);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void setTableDataTableCell(TableData *destination, double value, unsigned int columnIndex, unsigned int rowIndex, const unsigned int cellIndex, const unsigned int tableDataCount);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void setTableDataCounterPart(const TableData *source, const unsigned int sourceDataTableIndex, TableDataCell *destination);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void setTableDataCellPreviousNextCells(const TableData *tableData);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void clearTableData(TableData *tableData);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
unsigned int getTableDataTableCellIndex(const TableData *tableData, const unsigned int fromIndex, const unsigned int rowIndex, const unsigned int columnIndex);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
unsigned int getTableDataTableCellIndexByColumnIndex(const TableData *tableData, const unsigned int fromIndex, const unsigned int columnIndex);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
unsigned int getTableDataTableCellIndexByColumnIndexLast(const TableData *tableData, const unsigned int columnIndex);

void host_clearTableData(TableData *tableData);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
unsigned int findLargestColumnIndex(const TableData *tableData);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
bool isValidPreviousNextIndex(const TableData *source, const int previousIndex);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void clearTableDataCounterPart(TableData *tableData);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
unsigned int getMaxRowCountFromTableData(const TableData *tableData);

#endif // TABLE_DATA_H

