#ifndef readwritematrixtocsv
#define readwritematrixtocsv

#include <iostream>
#include <fstream>
#include "typedefs.h"
 
using namespace std;

namespace neuralnetworkfirstprinciples {

    void write_matrix_to_file(string fileName, Matrix  matrix);
    shared_ptr<Matrix> read_matrix_from_file(string fileToOpen);

    void write_parameters(vector<shared_ptr<Matrix>> weights, vector<shared_ptr<ColVector>> constants, string path, string file_name_prefix);
    void read_parameters(vector<shared_ptr<Matrix>>& weights, vector<shared_ptr<ColVector>>& constants, string path, string file_name_prefix);
}

#endif