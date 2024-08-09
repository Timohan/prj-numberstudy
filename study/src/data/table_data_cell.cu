/**
 * @file table_data_cell.cu
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief Contains host and device functions for table cells
 * 
 * @copyright Copyright (c) 2024
 */
#include "table_data_cell.h"
#include <stdio.h>

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief fill cell information into cell
 * 
 * @param value value of cell
 * @param columnIndex column index of cell
 * @param rowIndex row index of cell
 * @param destination [in/out] information (values) are filled to this cell
 * @param tableDataCount table count
 */
void setTableDataCell(double value, unsigned int columnIndex, unsigned int rowIndex, TableDataCell *destination, unsigned int tableDataCount)
{
    destination->m_value = value;
    destination->m_columnIndex = columnIndex;
    destination->m_rowIndex = rowIndex;
    destination->m_nextTableDataCell.m_initCompleted = false;
    destination->m_previousTableDataCell.m_initCompleted = false;
    destination->m_listCounterPartCount = tableDataCount;

    if (tableDataCount) {
        destination->m_listCounterPart = new TableDataCounterPart[tableDataCount];
        for (unsigned int i=0;i<tableDataCount;i++) {
            destination->m_listCounterPart[i].m_initCompleted = false;
        }
    }
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief set counter part cell to this cell
 * 
 * @param destination [in/out] counter part cell will be filled to this cell
 * @param toCounterPartIndex counter part index of cell
 * @param origin counter part cell to be filled into destination
 */
void setTableDataCellCounterPart(TableDataCell *destination, unsigned int toCounterPartIndex, const TableDataCell *counterPartCell)
{
    destination->m_listCounterPart[toCounterPartIndex].m_counterPart = counterPartCell;
    destination->m_listCounterPart[toCounterPartIndex].m_initCompleted = true;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief get previous (or next) cell
 * 
 * @param origin origin cell
 * @param previousIndex previous (or next) index,
 * -1 means previous cell
 * -2 means previous cell of previous cell
 * +1 means next cell...
 * @return const TableDataCell* pointer of previous (or next) cell
 * if nullptr, then the previous (or next) cell doesn't exists
 */
const TableDataCell *getPreviousNextCell(const TableDataCell *origin, int previousIndex)
{
    if (previousIndex == 0) {
        return origin;
    }

    const TableDataCell *tmp = origin;
    if (previousIndex < 0) {
        previousIndex = previousIndex*-1;
        for (int i=0;i<previousIndex;i++) {
            if (!tmp->m_previousTableDataCell.m_initCompleted) {
                return nullptr;
            }
            tmp = tmp->m_previousTableDataCell.m_counterPart;
            if (!tmp) {
                return tmp;
            }
        }
        return tmp;
    }
    for (int i=0;i<previousIndex;i++) {
        if (!tmp->m_nextTableDataCell.m_initCompleted) {
            return nullptr;
        }

        tmp = tmp->m_nextTableDataCell.m_counterPart;
        if (!tmp) {
            return tmp;
        }
    }
    return tmp;
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief clear pointers of counter parts list
 * 
 * @param cell [in/out] pointers are cleared here
 */
void clearTableDataCellCounterPart(TableDataCell *cell)
{
    if (cell->m_listCounterPartCount) {
        for (unsigned int i=0;i<cell->m_listCounterPartCount;i++) {
            cell->m_listCounterPart[i].m_counterPart = nullptr;
            cell->m_listCounterPart[i].m_initCompleted = false;
        }
    }
}

#ifdef CUDA_COMPILE
__host__ __device__
#endif
/**
 * @brief delete points of table 
 * 
 * @param cell [in/out] pointers are deleted here
 */
void clearTableDataCell(TableDataCell *cell)
{
    if (cell->m_listCounterPartCount) {
        delete[] cell->m_listCounterPart;
        cell->m_listCounterPart = nullptr;
        cell->m_listCounterPartCount = 0;
    }
}
