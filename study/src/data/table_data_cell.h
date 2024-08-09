/**
 * @file table_data_cell.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief Contains host and device functions for table cells
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef TABLE_DATA_CELL_H
#define TABLE_DATA_CELL_H

struct TableDataCell;

struct TableDataCounterPart
{
    bool m_initCompleted = false;
    const TableDataCell *m_counterPart = nullptr;
};

struct TableDataCell
{
    TableDataCounterPart *m_listCounterPart = nullptr;
    unsigned int m_listCounterPartCount = 0;

    TableDataCounterPart m_previousTableDataCell;
    TableDataCounterPart m_nextTableDataCell;

    unsigned int m_columnIndex = 0;
    unsigned int m_rowIndex = 0;

    double m_value;
};

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void setTableDataCell(double value, unsigned int columnIndex, unsigned int rowIndex, TableDataCell *destination, unsigned int tableDataCount);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void setTableDataCellCounterPart(TableDataCell *destination, unsigned int toCounterPartIndex, const TableDataCell *counterPartCell);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
const TableDataCell *getPreviousNextCell(const TableDataCell *origin, int previousIndex);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void clearTableDataCellCounterPart(TableDataCell *cell);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
void clearTableDataCell(TableDataCell *cell);

#endif // TABLE_DATA_H

