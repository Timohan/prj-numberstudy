/**
 * @file generate_matrix_lines.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief generate matrix line for study
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef GENERATE_MATRIX_LINES_H
#define GENERATE_MATRIX_LINES_H

#include "../data/list_table_data.h"
#include "../define_values.h"
#include "../matrix/cuda_matrix4xX.h"

#ifdef CUDA_COMPILE
__host__ __device__
#endif
bool generateMatrixLineValueFromCell(const int bestCounterPartMultiplier[MAX_COUNTER_PART_INDEX_COUNT+1],
                                     const int *previousIndex,
                                     const int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                                     CudaMatrix4xX::Matrix4xX *matrixLines,
                                     const unsigned int matrixLineColumnIndex,
                                     const TableDataCell *tableDataCell,
                                     const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT]);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
bool isAcceptableGenerating(const ListTableData *list,
                            const int counterPartIndex,
                            const int resultColumnIndex,
                            const int previousIndex,
                            const CounterPartMathType listCounterPartMathType);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
bool generateMatrixLine(const int bestCounterPartMultipliersPrimary[MAX_MATRIX_COLUMS],
                        const int bestCounterPartMultipliersCounter[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT],
                        const int *previousIndex,
                        const int listCounterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                        CudaMatrix4xX::Matrix4xX *matrixLines,
                        const TableData *tableData,
                        const unsigned int currentTableCellIndex,
                        const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT]);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
bool generateMatrixLines(const ListTableData *list,
                         const int bestCounterPartMultipliersPrimary[MAX_MATRIX_COLUMS],
                         const int bestCounterPartMultipliersCounter[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT],
                         const int previousIndex[MAX_MATRIX_COLUMS],
                         const int resultColumnIndex,
                         const int counterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                         CudaMatrix4xX::Matrix4xX *matrixLines,
                         const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT]);

#endif // GENERATE_MATRIX_LINES_H
