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
#include "search_part_index/unittest_search_part_index.h"

/*!
 * \brief main
 * \return
 */
int main(int argc , char *argv[])
{
    (void)argc;
    (void)argv;
    printf("cuda-tablestudy unit tests begin\n");
    printf("UnitSearchPartIndex: %s\n", UnitSearchPartIndex::search() ? "Ok" : "Failed");
    return 0;
}
