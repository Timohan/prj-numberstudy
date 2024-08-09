/**
 * @file options.h
 * @author  Timo Hannukkala <timohannukkala@hotmail.com>
 * @brief parses and set app launch params
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef CPP_OPTIONS_H
#define CPP_OPTIONS_H

#include <stdint.h>
#include <cstddef>
#include <vector>
#include <string>

class Options
{
public:
    bool set(int argc, char *argv[]);

    const int *getResultColumns();
    size_t getResultColumnsCount();

    bool getStudy();
    std::string getStudyResultFile();
    std::string getTableFile();

    void printHelp();
private:
    bool m_study = true;
    std::string m_studyResultFile = "result.dat";
    std::string m_tableFile = "";
    std::vector<int> m_listResultColumns;

};

#endif // CPP_OPTIONS_H
