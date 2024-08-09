/**
 * @file list_table_data.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief Contains list table data functionality
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef LIST_TABLE_DATA_H
#define LIST_TABLE_DATA_H

#include "table_data.h"
#include "table_data_cell.h"

struct ListTableData
{
    TableData *m_listTableData = nullptr;
    unsigned int m_listTableDataCount = 0;
};

#ifdef CUDA_COMPILE
__global__
#endif
void initListTableData(ListTableData *list, unsigned int tableDataCount);

#ifdef CUDA_COMPILE
__global__
#endif
void clearListTableData(ListTableData *list);

#ifdef CUDA_COMPILE
__global__
#endif
void setListTableDataTable(ListTableData *list, const TableData data, const unsigned tableIndex, unsigned int counterPartCount);

#ifdef CUDA_COMPILE
__global__
#endif
void setListTableDataTableCell(ListTableData *list, double value, unsigned int columnIndex, unsigned int rowIndex, const unsigned int tableIndex, const unsigned int cellIndex, unsigned int counterPartCount);

#ifdef CUDA_COMPILE
__global__
#endif
void setListTableDataCounterParts(ListTableData *list, const unsigned int destinationTableDataIndex);

#ifdef CUDA_COMPILE
__global__
#endif
void setListTableDataCellPreviousNextCells(ListTableData *list);

#ifdef CUDA_COMPILE
__host__
#endif
void host_initListTableData(ListTableData *list, unsigned int tableDataCount);

#ifdef CUDA_COMPILE
__host__
#endif
void host_clearListTableData(ListTableData *list);

#ifdef CUDA_COMPILE
__host__
#endif
void host_setListTableDataTable(ListTableData *list, const TableData *data, const unsigned tableIndex, unsigned int counterPartCount);

#ifdef CUDA_COMPILE
__host__
#endif
void host_setListTableDataTableCell(ListTableData *list, double value, unsigned int columnIndex, unsigned int rowIndex, const unsigned int tableIndex, const unsigned int cellIndex, unsigned int counterPartCount);

#ifdef CUDA_COMPILE
__host__
#endif
void host_setListTableDataCounterParts(ListTableData *list, const unsigned int destinationTableDataIndex);

#ifdef CUDA_COMPILE
__host__
#endif
void host_setListTableDataCellPreviousNextCells(ListTableData *list);

#ifdef CUDA_COMPILE
__host__
#endif
void host_generateAcceptableTableData(ListTableData *list, const int resultColumnIndex);

#ifdef CUDA_COMPILE
__global__
#endif
void generateAcceptableTableData(ListTableData *list, const int resultColumnIndex);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
unsigned int getMaxRowCountFromListTableData(const ListTableData *list);

#endif // LIST_TABLE_DATA_H

