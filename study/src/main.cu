/*!
 * \file
 * \brief file main.cpp
 *
*
 * Copyright of Timo Hannukkala. All rights reserved.
 *
 * \author Timo Hannukkala <timohannukkala@hotmail.com>
 */

#include <cstdlib>
#include <cstdio>
#include <time.h>
#ifndef CUDA_COMPILE
#include <cstdlib>
#include <cstring>
#else
#include <cuda.h>
#endif
#include "define_values.h"
#include "data/list_table_data.h"
#include "data/table_data.h"
#include "data/table_data_cell.h"
#include "loader/data_table_loader.h"
#include "loader/best_result_storage.h"
#include "calculate/calculate_results.h"
#include "study/study_finetune_primary_value.h"
#include "study/study_primary_value.h"
#include "study/study_find_new_counter_part.h"
#include "study/study_finetune_counter_part_value.h"
#include "options/options.h"
#include "common/time_difference.h"

#ifndef CUDA_BLOCK_NUM
#define CUDA_BLOCK_NUM 1
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1
#endif

#include "macros.h"

/*!
 * \brief main
 * \return
 */
int main(int argc , char *argv[])
{
    Options options;
    if (!options.set(argc, argv)) {
        options.printHelp();
        return 1;
    }

    if (options.getStudy()) {
        printf("Loading table numbers for study\n");
    } else {
        printf("Loading table numbers for calculate\n");
    }
    TimeDifference timeDifference;
    timeDifference.resetTimer();
    DataTableLoader m_dataTableLoader;
    m_dataTableLoader.load(options.getTableFile());
    uint64_t i;
    unsigned int cellIndex;
    BestResultStorage m_bestResult;

    const int *listResultColumnIndex = options.getResultColumns();
    int listResultColumnIndexCount = static_cast<int>(options.getResultColumnsCount());

    if (listResultColumnIndexCount > 0) {
        host_generateAcceptableTableData(m_dataTableLoader.getListTableData(), listResultColumnIndex[0]);
    }

    if (!options.getStudy()) {
        calculateResults(m_dataTableLoader, m_bestResult, options.getStudyResultFile());
        return 0;
    }

    ListTableData *d_listTableData;
    printf("Set tables for study\n");
#ifdef CUDA_COMPILE
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, NVIDIA_CUDA_HEAP_SIZE);
#endif
    PRJ_ALLOC(d_listTableData, ListTableData, 1)
    PRJ_FUNC_CALL_SINGLE(initListTableData, d_listTableData, m_dataTableLoader.getListTableData()->m_listTableDataCount);
    PRJ_CUDA_WAIT();
    for (i=0;i<m_dataTableLoader.getListTableData()->m_listTableDataCount;i++) {
        PRJ_FUNC_CALL_SINGLE(setListTableDataTable, d_listTableData, m_dataTableLoader.getListTableData()->m_listTableData[i], i, m_dataTableLoader.getListTableData()->m_listTableDataCount);
        for (cellIndex=0;cellIndex<m_dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCellCount;cellIndex++) {
            PRJ_FUNC_CALL_SINGLE(setListTableDataTableCell,
                d_listTableData,
                m_dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_value,
                m_dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_columnIndex,
                m_dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_rowIndex, i,
                cellIndex, m_dataTableLoader.getListTableData()->m_listTableDataCount);
        }
        PRJ_CUDA_WAIT();
    }

    for (i=0;i<m_dataTableLoader.getListTableData()->m_listTableDataCount;i++) {
        PRJ_FUNC_CALL_SINGLE(setListTableDataCounterParts, d_listTableData,  i);
    }
    PRJ_CUDA_WAIT();
    PRJ_FUNC_CALL_SINGLE(setListTableDataCellPreviousNextCells, d_listTableData);
    PRJ_CUDA_WAIT();

    PRJ_FUNC_CALL_SINGLE(generateAcceptableTableData, d_listTableData, listResultColumnIndex[0]);
    PRJ_CUDA_WAIT();
    printf("Study first primary values, Time: %lf\n", timeDifference.elapsedTimeFromBegin());

    StudyPrimaryValue::study(d_listTableData, listResultColumnIndex, listResultColumnIndexCount, &m_bestResult);

    for (int counterPartIndexPosition=0;counterPartIndexPosition<MAX_COUNTER_PART_INDEX_COUNT;counterPartIndexPosition++) {
        printf("Study searching counter part %d (max: %d), Time: %lf\n", counterPartIndexPosition+1, MAX_COUNTER_PART_INDEX_COUNT, timeDifference.elapsedTimeFromBegin());
        if (!StudyFindNewCounterPart::study(d_listTableData, listResultColumnIndex, listResultColumnIndexCount, &m_bestResult,
                                 m_dataTableLoader.getListTableData()->m_listTableDataCount)) {
            break;
        }
        PRJ_CUDA_WAIT();
    }

    StudyFinetuneCounterPartValue::study(d_listTableData, listResultColumnIndex, listResultColumnIndexCount, &m_bestResult, timeDifference);
    printf("Study searching primary value finetunes, Time: %lf\n", timeDifference.elapsedTimeFromBegin());
    StudyFinetunePrimaryValue::study(d_listTableData, listResultColumnIndex, listResultColumnIndexCount, &m_bestResult);

    PRJ_CUDA_WAIT();
    PRJ_FUNC_CALL_SINGLE(clearListTableData, d_listTableData);
    PRJ_FREE(d_listTableData)
    m_bestResult.save(options.getStudyResultFile());

    return 0;
}

