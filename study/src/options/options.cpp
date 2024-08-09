/**
 * @file options.cpp
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief parses and set app launch params
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "options.h"
#include <stdio.h>
#include <cstring>
#include <algorithm>

/**
 * @brief parses options
 * 
 * @param argc 
 * @param argv 
 * @return true 
 * @return false 
 */
bool Options::set(int argc , char *argv[])
{
    int i;
    for (i=1;i<argc;i++) {
        if (strncmp(argv[i], "--calc", 6) == 0) {
            m_study = false;
            continue;
        }
        if (strncmp(argv[i], "--study", 7) == 0) {
            m_study = true;
            continue;
        }
        if (strncmp(argv[i], "--result", 8) == 0) {
            if (i + 1 == argc) {
                return false;
            }
            i++;
            m_studyResultFile = argv[i];
            continue;
        }
        if (strncmp(argv[i], "--table", 7) == 0) {
            if (i + 1 == argc) {
                return false;
            }
            i++;
            m_tableFile = argv[i];
            continue;
        }
        const char *tmp = argv[i];
        if (tmp[0] >= '0' && tmp[0] <= '9') {
            if(std::find(m_listResultColumns.begin(), m_listResultColumns.end(), atoi(tmp)) == m_listResultColumns.end()) {
                m_listResultColumns.push_back(atoi(tmp));
            }
        }
    }
    if (m_tableFile.empty()) {
        return false;
    }
    if (!m_study) {
        return true;
    }
    return !m_listResultColumns.empty();
}

/**
 * @brief print help
 * 
 */
void Options::printHelp()
{
    printf("Options:\n");
    printf("--calc          - calculate (no study)\n");
    printf("--study         - study (default)\n");
    printf("--result [file] - result file of study (default: result.dat)\n");
    printf("--table [file]  - table file for study and calc (mandatory)\n");
    printf("<any number>    - column result, if multiple numbers, first number is for column result, other numbers are for secondary helpers (at least one is mandatory in study)\n");
}

/**
 * @brief
 * 
 * @return const int* list of columns, first column is the primary column
 */
const int *Options::getResultColumns()
{
    return m_listResultColumns.data();
}

/**
 * @brief column count for study
 * 
 * @return size_t 
 */
size_t Options::getResultColumnsCount()
{
    return m_listResultColumns.size();
}

/**
 * @brief is studying of not
 * 
 * @return true if studying and setting new result file
 * @return false if calculating from result file
 */
bool Options::getStudy()
{
    return m_study;
}

/**
 * @brief filename of result file
 * 
 * @return std::string 
 */
std::string Options::getStudyResultFile()
{
    return m_studyResultFile;
}

/**
 * @brief filename of tables that contains values
 * for study and calculate
 * 
 * @return std::string 
 */
std::string Options::getTableFile()
{
    return m_tableFile;
}
