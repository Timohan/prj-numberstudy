/**
 * @file calculate_results.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief calculates results
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef CALCULATE_RESULTS_H
#define CALCULATE_RESULTS_H

#include "../define_values.h"
#include "../data/list_table_data.h"
#include "../data/table_data.h"
#include "../data/table_data_cell.h"
#include "../loader/data_table_loader.h"
#include "../loader/best_result_storage.h"

void calculateResults(
    DataTableLoader &dataTableLoader,
    BestResultStorage &bestResult,
    const std::string &resultFile);

#endif // CALCULATE_RESULTS_H
