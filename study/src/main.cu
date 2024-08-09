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

#ifndef CUDA_BLOCK_NUM
#define CUDA_BLOCK_NUM 1
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1
#endif

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
    cudaMalloc((void**)&d_listTableData, sizeof(ListTableData));
    initListTableData<<<1, 1>>>(d_listTableData, m_dataTableLoader.getListTableData()->m_listTableDataCount);
    cudaDeviceSynchronize();
#else
    d_listTableData = new ListTableData;
    host_initListTableData(d_listTableData, m_dataTableLoader.getListTableData()->m_listTableDataCount);
#endif
    for (i=0;i<m_dataTableLoader.getListTableData()->m_listTableDataCount;i++) {
#ifdef CUDA_COMPILE
        setListTableDataTable<<<1, 1>>>(d_listTableData, m_dataTableLoader.getListTableData()->m_listTableData[i], i, m_dataTableLoader.getListTableData()->m_listTableDataCount);
#else
        host_setListTableDataTable(d_listTableData, &m_dataTableLoader.getListTableData()->m_listTableData[i], i, m_dataTableLoader.getListTableData()->m_listTableDataCount);
#endif
        for (cellIndex=0;cellIndex<m_dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCellCount;cellIndex++) {
#ifdef CUDA_COMPILE
            setListTableDataTableCell<<<1, 1>>>(
                d_listTableData,
                m_dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_value,
                m_dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_columnIndex,
                m_dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_rowIndex, i, 
                cellIndex, m_dataTableLoader.getListTableData()->m_listTableDataCount);
#else
            host_setListTableDataTableCell(
                d_listTableData,
                m_dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_value,
                m_dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_columnIndex,
                m_dataTableLoader.getListTableData()->m_listTableData[i].m_listTableCell[cellIndex].m_rowIndex, i, 
                cellIndex, m_dataTableLoader.getListTableData()->m_listTableDataCount);
#endif
        }
#ifdef CUDA_COMPILE
        cudaDeviceSynchronize();
#endif
    }

#ifdef CUDA_COMPILE
    for (i=0;i<m_dataTableLoader.getListTableData()->m_listTableDataCount;i++) {
        setListTableDataCounterParts<<<1, 1>>>(d_listTableData,  i);
    }
    cudaDeviceSynchronize();
    setListTableDataCellPreviousNextCells<<<1, 1>>>(d_listTableData);
    cudaDeviceSynchronize();

    generateAcceptableTableData<<<1, 1>>>(d_listTableData, listResultColumnIndex[0]);
    cudaDeviceSynchronize();
#else
    for (i=0;i<m_dataTableLoader.getListTableData()->m_listTableDataCount;i++) {
        host_setListTableDataCounterParts(d_listTableData,  i);
    }
    host_setListTableDataCellPreviousNextCells(d_listTableData);
    host_generateAcceptableTableData(d_listTableData, listResultColumnIndex[0]);
#endif
    printf("Study first primary values\n");

    StudyPrimaryValue::study(d_listTableData, listResultColumnIndex, listResultColumnIndexCount, &m_bestResult);

    for (int counterPartIndexPosition=0;counterPartIndexPosition<MAX_COUNTER_PART_INDEX_COUNT;counterPartIndexPosition++) {
        printf("Study searching counter part %d (max: %d)\n", counterPartIndexPosition+1, MAX_COUNTER_PART_INDEX_COUNT);
        if (!StudyFindNewCounterPart::study(d_listTableData, listResultColumnIndex, listResultColumnIndexCount, &m_bestResult,
                                 m_dataTableLoader.getListTableData()->m_listTableDataCount)) {
            break;
        }
#ifdef CUDA_COMPILE
        cudaDeviceSynchronize();
#endif
    }

    StudyFinetuneCounterPartValue::study(d_listTableData, listResultColumnIndex, listResultColumnIndexCount, &m_bestResult);
    printf("Study searching primary value finetunes\n");
    StudyFinetunePrimaryValue::study(d_listTableData, listResultColumnIndex, listResultColumnIndexCount, &m_bestResult);

#ifdef CUDA_COMPILE
    cudaDeviceSynchronize();
    clearListTableData<<<1, 1>>>(d_listTableData);
    cudaFree(d_listTableData);
#else
    host_clearListTableData(d_listTableData);
    delete d_listTableData;
#endif
    m_bestResult.save(options.getStudyResultFile());

    return 0;
}

