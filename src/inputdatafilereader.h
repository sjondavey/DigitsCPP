#ifndef inputfiledatareader
#define inputfiledatareader

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include "./neuralnetwork.h"
#include <memory>

using namespace std;

namespace neuralnetworkfirstprinciples {
    // Takes a number from 0 to 9 and converts it to a column vector where all the elements are zero except
    // the one corresponding to the input number, which is set to 1
    shared_ptr<ColVector> convertNumberToVector(size_t number);

    // Normalizes the input data by dividing by 256
    void read_digit_data(string filename, 
                         vector<shared_ptr<ColVector>>& labelAsVectors, 
                         vector<shared_ptr<ColVector>>& digitisedNumbers);

    // Used for vectorized version of the function. Each column of the 'digitisedNumbers' matrix is an
    // observation and the corresponding column of 'labels' is the label converted into an vector
    // of length 10 with all elements set to zero except the one corresponding to the label value which 
    // is set to 1.
    // Data is normalized in this method by dividing all input data by 256
    // This method assumes there is a header row 
    void read_digit_data(string filename, 
                         shared_ptr<Matrix>& labels, 
                         shared_ptr<Matrix>& data);

}

#endif
