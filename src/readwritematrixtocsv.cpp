#include "readwritematrixtocsv.h"

namespace neuralnetworkfirstprinciples {

    void write_parameters(vector<shared_ptr<Matrix>> weights, vector<shared_ptr<ColVector>> constants, string path, string file_name_prefix)
    {
        for (size_t i = 0; i < weights.size(); ++i)
        {
            string weights_file_name = path + file_name_prefix + "_W" + to_string(i) + ".csv";
            write_matrix_to_file(weights_file_name, (*weights[i]));

            string constants_file_name = path + file_name_prefix + "_b" + to_string(i) + ".csv";
            write_matrix_to_file(constants_file_name, (*constants[i]));
        }
        return;
    }

    void read_parameters(vector<shared_ptr<Matrix>>& weights, vector<shared_ptr<ColVector>>& constants, string path, string file_name_prefix)
    {
        size_t i = 0;
        string weights_file_name = path + file_name_prefix + "_W0.csv";
        string constants_file_name = path + file_name_prefix + "_b0.csv";

        ifstream f(weights_file_name.c_str());
        bool file_exists = f.good();
        weights = vector<shared_ptr<Matrix>>(0);
        constants = vector<shared_ptr<ColVector>>(0);
        while (file_exists)
        {
            weights.push_back(read_matrix_from_file(weights_file_name));
            shared_ptr<Matrix> tmp = read_matrix_from_file(constants_file_name);
            shared_ptr<ColVector> tmp_as_col;
            if (tmp->cols() != 1) {
                throw invalid_argument("Constants are not column vectors");
            }
            else {
                tmp_as_col = (make_shared<ColVector>)((ColVector) *tmp);
            }
            constants.push_back(tmp_as_col);

            ++i;
            weights_file_name = path + file_name_prefix + "_W" + to_string(i) + ".csv";
            constants_file_name = path + file_name_prefix + "_b" + to_string(i) + ".csv";
            f = ifstream(weights_file_name.c_str());
            file_exists = f.good();
        }
    }


    void write_matrix_to_file(string fileName, Matrix  matrix)
    {
        //https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
        const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", "\n");
    
        ofstream file(fileName);
        if (file.is_open())
        {
            file << matrix.format(CSVFormat);
            file.close();
        }
    }

    shared_ptr<Matrix> read_matrix_from_file(string fileToOpen)
    {
        // the input is the file: "fileToOpen.csv":
        // a,b,c
        // d,e,f
        // This function converts input file data into the Eigen matrix format
    
        // the matrix entries are stored in this variable row-wise. For example if we have the matrix:
        // M=[a b c 
        //    d e f]
        // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
        // later on, this vector is mapped into the Eigen matrix format
        vector<Scalar> matrixEntries;
        // in this object we store the data from the matrix
        ifstream matrixDataFile(fileToOpen);
        // this variable is used to store the row of the matrix that contains commas 
        string matrixRowString;    
        // this variable is used to store the matrix entry;
        string matrixEntry;    
        // this variable is used to track the number of rows
        int matrixRowNumber = 0;
        while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
        {
            stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.
            while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
            {
                matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
            }
            matrixRowNumber++; //update the column numbers
        }
        // here we convert the vector variable into the matrix and return the resulting object, 
        // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
        int number_of_columns = matrixEntries.size() / matrixRowNumber;
        return make_shared<Matrix>(Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, number_of_columns));
    
    }
}