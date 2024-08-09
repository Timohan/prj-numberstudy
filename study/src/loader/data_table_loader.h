/**
 * @file data_table_loader.h
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief table data loader
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef DATA_TABLE_LOADER_H_
#define DATA_TABLE_LOADER_H_

#include <vector>
#include <string>
#include <fstream>
#include "../data/list_table_data.h"
struct TableDataCell;

struct TemporaryDataCell
{
    unsigned int m_columnIndex;
    unsigned int m_rowIndex;
    double m_value;
};

struct TemporaryDataCellList
{
    std::vector<TemporaryDataCell> m_listCell;
};

class DataTableLoader
{
public:
    bool load(const std::string &tableFile);
    ~DataTableLoader();
    ListTableData *getListTableData();
    std::string getTableType(unsigned int i);

private:
    ListTableData m_listTableData;
    std::vector<std::string> m_listDataTableType;

    static unsigned int getTableCount(std::ifstream &infile);
    static void removeNonRequiredCharactersFromEnd(std::string &line);
    static void removeNonRequiredCharactersFromStart(std::string &line);
    static unsigned int getNumberAfterText(const std::string &line, const std::string &textBeforeNumber);
    static void loadCell(std::string &line, unsigned int columnIndex, unsigned int rowIndex, std::vector<TemporaryDataCell> &listCell);
    void fillCellsToListTableData(const std::vector<TemporaryDataCell> &listCell, unsigned int tableIndex, unsigned int tableCount);
};

#endif // DATA_TABLE_LOADER_H_