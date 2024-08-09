/**
 * @file generate_matrix_lines_calculate.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief generates matrix lines for calculate
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef GENERATE_MATRIX_LINES_CALCULATE_H
#define GENERATE_MATRIX_LINES_CALCULATE_H

#include "../data/list_table_data.h"
#include "../define_values.h"
#include "../matrix/cuda_matrix4xX.h"

#ifdef CUDA_COMPILE
__host__ __device__
#endif
bool generateMatrixLinesCalculate(const ListTableData *list,
                         const int bestCounterPartMultipliersPrimary[MAX_MATRIX_COLUMS],
                         const int bestCounterPartMultipliersCounter[MAX_MATRIX_COLUMS*MAX_COUNTER_PART_INDEX_COUNT],
                         const int *previousIndex,
                         const int resultColumnIndex,
                         const int counterPartIndex[MAX_COUNTER_PART_INDEX_COUNT],
                         CudaMatrix4xX::Matrix4xX *matrixLines,
                         const CounterPartMathType listCounterPartMathType[MAX_COUNTER_PART_INDEX_COUNT]);

#endif // GENERATE_MATRIX_LINES_CALCULATE_H
