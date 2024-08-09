/**
 * @file data_table_loader.cpp
 * @author Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief table data loader
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "data_table_loader.h"
#include <iostream>
#include <cstring>

DataTableLoader::~DataTableLoader()
{
    host_clearListTableData(&m_listTableData);
}

ListTableData *DataTableLoader::getListTableData()
{
    return &m_listTableData;
}

/**
 * @brief load table data value
 * 
 * @param tableFile text file that contains table values
 * @return true if loading was success.
 * @return false 
 */
bool DataTableLoader::load(const std::string &tableFile)
{
    std::string line;
    std::vector<TemporaryDataCellList> listTemporaryDataCell;

    unsigned int columnIndex = 0;
    unsigned int rowIndex = 0;
    unsigned int tableIndex = __UINT32_MAX__;
    unsigned int i;

    std::ifstream infile;
    infile.open(tableFile);
    if (!infile.is_open()) {
        std::cout << "File open failed: " << tableFile << std::endl;
        return false;
    }

    infile.clear();
    infile.seekg(0, std::ios::beg);
    while (getline(infile, line)) {
        removeNonRequiredCharactersFromStart(line);
        removeNonRequiredCharactersFromEnd(line);
        if (line.empty() || line.front() == '#') {
            continue;
        }
        if (line.compare(0, 5, "Type:") == 0) {
            listTemporaryDataCell.push_back(TemporaryDataCellList());
            if (tableIndex == __UINT32_MAX__) {
                tableIndex = 0;
            } else {
                tableIndex++;
            }
            line = line.substr(5);
            removeNonRequiredCharactersFromStart(line);
            if (line.empty()) {
                line = " ";
            }
            m_listDataTableType.push_back(line);
            columnIndex = 0;
            rowIndex = 0;
            continue;
        }
        if (line.compare(0, strlen("Start Column:"), "Start Column:") == 0) {
            removeNonRequiredCharactersFromStart(line);
            columnIndex = getNumberAfterText(line, "Start Column:");
            continue;
        }
        if (line.compare(0, strlen("Start Row:"), "Start Row:") == 0) {
            removeNonRequiredCharactersFromStart(line);
            rowIndex = getNumberAfterText(line, "Start Row:");
            continue;
        }
        if ( (line.front() >= '0' && line.front() <= '9') || line.front() == '-') {
            if (tableIndex == __UINT32_MAX__) {
                std::cout << "No Type: before numbers start" << tableFile << std::endl;
                infile.close();
                return false;
            }
            while (1) {
                removeNonRequiredCharactersFromStart(line);
                if (line.empty()) {
                    break;
                }

                loadCell(line, columnIndex, rowIndex, listTemporaryDataCell.back().m_listCell);
                columnIndex++;
            }
            columnIndex = 0;
            rowIndex++;
            continue;
        }
    }

    infile.close();

    host_initListTableData(&m_listTableData, listTemporaryDataCell.size());

    for (i=0;i<listTemporaryDataCell.size();i++) {
        fillCellsToListTableData(listTemporaryDataCell.at(i).m_listCell, i,
                                 listTemporaryDataCell.size());
    }


    for (i=0;i<getListTableData()->m_listTableDataCount;i++) {
        host_setListTableDataCounterParts(getListTableData(), i);
    }

    host_setListTableDataCellPreviousNextCells(getListTableData());

    return true;
}

/**
 * @brief finds first number from line and creates temporary table cell
 * 
 * @param line [in/out] finds first number from this line and also removes the number from line
 * @param columnIndex return cell will be for this column index
 * @param rowIndex  return cell will be for row index
 * @param ListCell new cell is added here
 */
void DataTableLoader::loadCell(std::string &line, unsigned int columnIndex, unsigned int rowIndex, std::vector<TemporaryDataCell> &listCell)
{
    TemporaryDataCell cell;
    cell.m_columnIndex = columnIndex;
    cell.m_rowIndex = rowIndex;
    cell.m_value = std::stod(line);
    listCell.push_back(cell);

    size_t nextIndex0 = line.find("\t");
    size_t nextIndex1 = line.find(" ");

    if (nextIndex0 == std::string::npos
        && nextIndex1 == std::string::npos) {
        line.clear();
        return;
    }

    if (nextIndex0 == std::string::npos) {
        line = line.substr(nextIndex1);
        removeNonRequiredCharactersFromStart(line);
        return;
    }
    if (nextIndex1 == std::string::npos) {
        line = line.substr(nextIndex0);
        removeNonRequiredCharactersFromStart(line);
        return;
    }

    if (nextIndex1 < nextIndex0) {
        line = line.substr(nextIndex1);
        removeNonRequiredCharactersFromStart(line);
        return;
    }

    line = line.substr(nextIndex0);
    removeNonRequiredCharactersFromStart(line);
}

/**
 * @brief get table count
 * 
 * @param infile file
 * @return unsigned int table count from the file
 */
unsigned int DataTableLoader::getTableCount(std::ifstream &infile)
{
    unsigned int ret = 0;
    std::string line; 
    while (getline(infile, line)) {
        if (line.compare(0, 5, "Type:") == 0) {
            ret++;
        }
    }
    return ret;
}

/**
 * @brief get unsigned int after the text textBeforeNumber
 * 
 * @param line line that contains textBeforeNumber and number
 * @param textBeforeNumber number comes after this string
 * @return unsigned int number
 */
unsigned int DataTableLoader::getNumberAfterText(const std::string &line, const std::string &textBeforeNumber)
{
    std::string numberText = line.substr(textBeforeNumber.size());
    removeNonRequiredCharactersFromStart(numberText);
    return std::stoul(numberText);
}

/**
 * @brief DataTableLoader::removeNonRequiredCharactersFromEnd
 * 
 * @param line [in/out] remove non required characters from the end of this line
 */
void DataTableLoader::removeNonRequiredCharactersFromEnd(std::string &line)
{
    while (1) {
        if (line.empty()) {
            break;
        }
        if (line.back() == '\n' || line.back() == '\r' || line.back() == ' ' || line.back() == '\t') {
            line.resize(line.size()-1);
            continue;
        }
        break;
    }
}

/**
 * @brief DataTableLoader::removeNonRequiredCharactersFromStart
 * 
 * @param line [in/out] remove non required characters from the start of this line
 */
void DataTableLoader::removeNonRequiredCharactersFromStart(std::string &line)
{
    while (1) {
        if (line.empty()) {
            break;
        }
        if (line.front() == '\n' || line.front() == '\r' || line.front() == ' ' || line.front() == '\t') {
            line = line.substr(1);
            continue;
        }
        break;
    }
}

/**
 * @brief creates new table data to m_listTableData
 * 
 * @param listCell list of cells to add to new table data of m_listTableData
 * @param tableIndex new table index
 * @param tableCount table count
 */
void DataTableLoader::fillCellsToListTableData(const std::vector<TemporaryDataCell> &listCell,
                                               unsigned int tableIndex,
                                               unsigned int tableCount)
{
    size_t i;
    TableData tableData;
    tableData.m_listTableCellCount = listCell.size();

    tableData.m_listTableCell = new TableDataCell[listCell.size()];
    for (i=0;i<listCell.size();i++) {
        tableData.m_listTableCell[i].m_value = listCell[i].m_value;
        tableData.m_listTableCell[i].m_columnIndex = listCell[i].m_columnIndex;
        tableData.m_listTableCell[i].m_rowIndex = listCell[i].m_rowIndex;
        tableData.m_listTableCell[i].m_listCounterPartCount = tableCount;
    }

    host_setListTableDataTable(&m_listTableData, &tableData, tableIndex, tableData.m_listTableCellCount);

    for (unsigned int cellIndex=0;cellIndex<tableData.m_listTableCellCount;cellIndex++) {
        host_setListTableDataTableCell(&m_listTableData,
            tableData.m_listTableCell[cellIndex].m_value,
            tableData.m_listTableCell[cellIndex].m_columnIndex,
            tableData.m_listTableCell[cellIndex].m_rowIndex, tableIndex, 
            cellIndex, tableCount);
    }
    host_clearTableData(&tableData);
}

/**
 * @brief get table type strings
 * 
 * @param i index of table strings
 * @return table type string of index
 */
std::string DataTableLoader::getTableType(unsigned int i)
{
    return m_listDataTableType.at(i);
}
