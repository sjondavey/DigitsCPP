#include "inputdatafilereader.h"

namespace neuralnetworkfirstprinciples {

    shared_ptr<ColVector> convertNumberToVector(size_t number)
    {
        shared_ptr<ColVector> column = make_shared<ColVector>(10);
        column->setZero();
        (*column)(number) = 1.0;
        return column;
    }

    // This assumes the file exists and no checks are performed here.
    void read_digit_data(string filename, vector<shared_ptr<ColVector>>& labelAsVectors, vector<shared_ptr<ColVector>>& digitisedNumbers)
    {
        digitisedNumbers.clear();
        labelAsVectors.clear();
        Scalar normalization_constant = 256.0;

        std::ifstream file(filename);
        std::string line, word;

        // read header line
        getline(file, line, '\n');
        std::stringstream ss(line);
        std::vector<string> parsed_vec;
        while (getline(ss, word, ',')) {
            parsed_vec.push_back(&word[0]);
        }
        size_t cols = parsed_vec.size();
        size_t data_length = cols - 1;

        size_t lines_read = 0;
        if (file.is_open()) {
            while (getline(file, line, '\n')) {
                ss = stringstream(line);

                // first entry is the label as an integer between 0 and 9
                getline(ss, word, ',');
                int number = std::stoi(&word[0]);
                if ((number < 0) || (number > 9))
                {
                    throw out_of_range("Labels need to be between 0 and 9 but we encountered " + word);
                }
                labelAsVectors.push_back(convertNumberToVector(number));


                digitisedNumbers.push_back(make_shared<ColVector>(data_length, 1));
                size_t i = 0;
                while (getline(ss, word, ',')) {
                    digitisedNumbers.back()->coeffRef(i) = Scalar(std::stof(&word[0]) / normalization_constant);
                    i++;
                }
                // string filename = "E:/Code/kaggle/digits/data/digitisednumber" + to_string(lines_read) + ".csv";
                // writeData(filename, (*digitisedNumbers.back()));
                lines_read +=1;
                if (lines_read % 1000 == 0)
                    cout << lines_read << " lines have been read" << endl;
            }
        }

    }


    void read_digit_data(string filename, 
                         shared_ptr<Matrix>& labels, 
                         shared_ptr<Matrix>& data)
    {
        vector<shared_ptr<ColVector>> labelAsVectors;
        vector<shared_ptr<ColVector>> digitisedNumbers;
        read_digit_data(filename, labelAsVectors, digitisedNumbers);

        assert(labelAsVectors.size() == digitisedNumbers.size());
        assert(labelAsVectors[0]->rows() == 10);
        assert(digitisedNumbers[0]->rows() == 784);

        labels = make_shared<Matrix>(10, labelAsVectors.size());
        data = make_shared<Matrix>(784, digitisedNumbers.size());
        for (size_t i = 0; i < labelAsVectors.size(); ++i)
        {
            labels->col(i) = (*labelAsVectors[i]);
            data->col(i) = (*digitisedNumbers[i]);
        }
    }

}

