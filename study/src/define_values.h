#ifndef DEFINE_VALUES_H__
#define DEFINE_VALUES_H__

/**
 * @brief nvidia cuda malloc heap size
 * 1073741824 = 1024*1024*1024
 */
#define NVIDIA_CUDA_HEAP_SIZE                 1073741824

/**
 * @brief nvidia gpu SM count
 * 
 */
#define NVIDIA_GPU_SM_COUNT                 15

/**
 * @brief how many counter part indexes are for the calculation (max)
 * for example, if you are calculating cpi (all), how many other table
 * values are taken to count, for example pmi and other cpi
 */
#define MAX_COUNTER_PART_INDEX_COUNT                   30

/**
 * @brief secondary's index's column's month, for example, if you are calculation inflation
 * and secondary index is "pmi manufacturing", and previous month is is -2
 *  (must be between MIN_PREVIOUS_NEXT_INDEX and MAX_PREVIOUS_NEXT_INDEX)
 * then pmi manufacuring values 2 previous month from inflation month
 */
#define MIN_PREVIOUS_NEXT_INDEX -3
#define MAX_PREVIOUS_NEXT_INDEX 1

/**
 * @brief sets how many rows must be on the the tables
 * that table data is allowed for taking into calculations
 */
#define MIN_ROW_LINE_COUNT_FOR_MAKE_MATRIX_CALC     7

/**
 * @brief primary rate multiplier step count
 * this value must be between 2-100
 */
#define PRIMARY_RATE_MULTIPLIER_STEP_COUNT 10

/**
 * @brief finetune rate min/max values, min value must be >= 1,
 * and max value must be <= 100
 */
#define PRIMARY_FINETUNE_RATE_FIND_MIN 1
#define PRIMARY_FINETUNE_RATE_FIND_MAX 100

/**
 * @brief counter part rate multiplier step count
 * this value must be between 2-100
 */
#define COUNTER_PART_RATE_MULTIPLIER_STEP_COUNT 10

/**
 * @brief default best value, do not change this
 */
#define DEFAULT_BEST_VALUE  10000000

/**
 * @brief cuda blocks and threads per block
 */
#define CUDA_BLOCK_NUM      (NVIDIA_GPU_SM_COUNT*32)
#define THREADS_PER_BLOCK   256

/**
 * @brief it's using 4x4 matrix
 * Do not change this
 */
#define MAX_MATRIX_COLUMS   4

/**
 * @brief default primary value multiplier
 */
#define DEFAULT_PRIMARY_VALUE_MULTIPLIER    50

typedef unsigned long long int uint64_cu;

enum CounterPartMathType
{
    CounterPartMathType_Plus_Normal = 0,
    CounterPartMathType_Plus_PreviousMinus,
    CounterPartMathType_Count
};

#endif