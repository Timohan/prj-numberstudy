# prj-numberstudy

prj-numberstudy is my personal experiment project to find economical values from number of spreadsheets. It does support CUDA using nvidia gpu and CPU compiles.

Using nvidia's GPU with CUDA is much faster way to find than CPU.

## Build program

### Development Environment with CUDA

CUDA libraries (Nvidia's drivers).

### Set SM Count in src/define_values.h

You can modify SM count on file src/define_values.h

Line: #define NVIDIA_GPU_SM_COUNT                 15

Change the number "15" to for your Nvidia's GPU SM count.  
Larger number is faster as long your GPU have enough SM count.

### CUDA Heap size

On large number of spreadsheets, you may also need increase the heap size

Line: "#define NVIDIA_CUDA_HEAP_SIZE                 1073741824"

### Build program with CUDA

cd study  
make  

### Build program with CPU (alternative, slower)

cd study  
./compile_x86.sh  

## Study (search value)

### Example spreadsheet (tables)

Spreadsheet (table) must have at least 8 rows (lines) and at least 6 columns of numbers for line. All lines should have equal number rows, except first row and last column can have less number of columns.

#### Spreadsheet documentation

"Type: " - Each table in spreadsheet starts with "Type: ", type is name of the table.  
"Start Column: <number>" - First column number of first row. For example, if you have normally 12 columns of numbers in table, but only 11 numbers in first row and first number is 2nd column, set "Start Column: 1".  
"Start Row: <number>" - First row number of table, if you have unequal number of rows in each tables, set start row so program knows the row index of first row.

Numbers of rows are then separated with '\\t' (tab) or '&nbsp;' (space). All numbers should be smaller than 10, and bigger than -10, because heavy internal calculation with multiply can set values too large.

Primary table, to search numbers for this table, is always at first.

#### Example files

tables/cpi_only.txt - this file contains only primary table  
tables/multiple_cpi.txt - this file contains multiple tables and first is always the primary table to find

### Study (search value of column)

#### Args

--study - search calculate values for column  
--calc  - calculate value for column after study is completed  
--result [file] - result file of study (default: result.dat)  
--table [file]  - table file for study and calc (mandatory)  
\<any number\> - column index to search, first column index is 0, so for example, if you want to search values for column 6 (7th column) use number 6. Other numbers are helper columns for calculate. At least one column is required for "study".

#### Example study
'./prj-numberstudy --study 6 --table ../tables/cpi_only.txt'

In this case, ../tables/cpi_only.txt file have only 6 numbers in last row. And it tries find values for 7th column(s).

'./prj-numberstudy --study 5 --table ../tables/multiple_cpi.txt --result result_multiple_cpi.dat'

In this case, ../tables/multiple_cpi.txt file have only 5 numbers in last row. And it tries find values for 6th column(s).

'./prj-numberstudy --study 5 4 --table ../tables/multiple_cpi.txt --result result_multiple_cpi2.dat'

In this case, ../tables/multiple_cpi.txt file have only 5 numbers in last row. And it tries find values for 6th column(s), but using using also previous columns as helper column. This might lead to more inaccurate results with smaller number of tables.

#### Calculate result for studied column(s)

'./prj-numberstudy --calc --table ../tables/cpi_only.txt'

Results:  
Result for this row: 1.763291 (Original result: 2.000000)  
Result for this row: 1.887342 (Original result: 2.000000)  
Result for this row: 0.200000 (Original result: 0.200000)  
Result for this row: 0.800000 (Original result: 0.800000)  
Result for this row: 1.700000 (Original result: 1.700000)  
Result for this row: 2.900000 (Original result: 2.900000)  
Result for this row: 1.544937 (Original result: 1.800000)  
Result for this row: 0.469620 (Original result: 1.000000)  
Result for this row: 4.743038 (Original result: 5.400000)  
Result for this row: 8.643671 (Original result: 8.500000)  
Result for this row: 3.231013 (Original result: 3.200000)  
Result for this row: 3.167722 (Original result: NEW)

First row had original result 2.000000, but with studied calculate is 1.763291. And last row had no 7th column, but studied math estimates it to be 3.167722.

---

'./prj-numberstudy --calc --table ../tables/multiple_cpi.txt --result result_multiple_cpi.dat'

Results:  
Result for this row: 1.000018 (Original result: 1.000000)  
\<snip\>  
Result for this row: 3.657517 (Original result: NEW)

First row had original result 1.000000, but with studied calculate is 1.000018. And last row had no 6th column, but studied math estimates it to be 3.657517.

---

'./prj-numberstudy --calc --table ../tables/multiple_cpi.txt --result result_multiple_cpi2.dat'

Results:  
Result for this row: 0.084545 (Original result: 0.100000)  
\<snip\>  
Result for this row: 2.899832 (Original result: NEW)

First row had original result 1.000000, but with studied calculate is 0.084545. And last row had no 6th column, but studied math estimates it to be 2.899832.


