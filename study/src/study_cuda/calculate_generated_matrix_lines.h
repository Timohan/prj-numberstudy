/**
 * @file calculate_generated_matrix_lines.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief calculate generated matrix lines
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef CALCULATE_GENERATED_MATRIX_LINES_H
#define CALCULATE_GENERATED_MATRIX_LINES_H

#include "../matrix/cuda_matrix4xX.h"

#ifdef CUDA_COMPILE
__host__ __device__
#endif
double calculateMatrix4xXLine(const CudaMatrix4x4::Matrix4x1 *line,
                              const CudaMatrix4xX::Matrix4xX *matrixLines,
                              const unsigned int baseMatrixLineIndex);

#ifdef CUDA_COMPILE
__host__ __device__
#endif
bool calculateGeneratedMatrixLines(double *localBestResult, double *globalBestResultMax,
                    const CudaMatrix4xX::ListMatrix4xX *listMatrixLines,
                    double calculatedDotProductValues[MAX_MATRIX_COLUMS]);

#endif // CALCULATE_GENERATED_MATRIX_LINES_H
