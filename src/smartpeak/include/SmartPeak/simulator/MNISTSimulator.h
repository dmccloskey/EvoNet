/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MNISTSIMULATOR_H
#define SMARTPEAK_MNISTSIMULATOR_H

#include <SmartPeak/simulator/DataSimulator.h>
#include <SmartPeak/core/Preprocessing.h>

namespace SmartPeak
{
  /**
    @brief A class to generate data using the MNIST data set
  */
  class MNISTSimulator: public DataSimulator
  {
public:
		int ReverseInt(int i)
		{
			unsigned char ch1, ch2, ch3, ch4;
			ch1 = i & 255;
			ch2 = (i >> 8) & 255;
			ch3 = (i >> 16) & 255;
			ch4 = (i >> 24) & 255;
			return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
		}

		/*
		@brief Read in the MNIST data set from an IDX file format.

		Output data for sample dimensions are the following:
			dim 0: sample
			dim 1: col-wise pixel intensity

		Output data for label dimensions are the following:
			dim 0: sample
			dim 1: class label

		See http://yann.lecun.com/exdb/mnist/ for a description of the data set and the file format

		@param[in] filename
		@param[in, out] data The tensor to hold the data
		@param[in] is_labels True if the file corresponds to class labels, False otherwise
		*/
		template<typename T>
		void ReadMNIST(const std::string& filename, Eigen::Tensor<T, 2>& data, const bool& is_labels)
		{
			// dims: sample, pixel intensity or sample, label
			// e.g., pixel data dims: 1000 x (28x28) (stored row-wise; returned col-wise)
			// e.g., label data dims: 1000 x 1

			// open up the file
			std::ifstream file(filename, std::ios::binary);
			if (file.is_open())
			{
				int magic_number = 0;
				int number_of_images = 0;
				int n_rows = 0;
				int n_cols = 0;

				// get the magic number
				file.read((char*)&magic_number, sizeof(magic_number));
				magic_number = ReverseInt(magic_number);

				// get the number of images
				file.read((char*)&number_of_images, sizeof(number_of_images));
				number_of_images = ReverseInt(number_of_images);
				if (number_of_images > data.dimension(0))
					number_of_images = data.dimension(0);

				// get the number of rows and cols
				if (!is_labels)
				{
					file.read((char*)&n_rows, sizeof(n_rows));
					n_rows = ReverseInt(n_rows);
					file.read((char*)&n_cols, sizeof(n_cols));
					n_cols = ReverseInt(n_cols);
				}
				else
				{
					n_rows = 1;
					n_cols = 1;
				}

				// get the actual data (read row-wise)
				for (int i = 0; i < number_of_images; ++i)
				{
					for (int r = 0; r < n_rows; ++r)
					{
						for (int c = 0; c < n_cols; ++c)
						{
							unsigned char temp = 0;
							file.read((char*)&temp, sizeof(temp));
							//data(i, (n_rows*r) + c) = (T)temp; // row-wise return
							data(i, (n_cols*c) + r) = (T)temp; // col-wise return
						}
					}
				}
			}
		}

		void readData(const std::string& filename_data, const std::string& filename_labels, const bool& is_training,
			const int& data_size, const int& input_size)
		{
			// Read input images [BUG FREE]
			Eigen::Tensor<float, 2> input_data(data_size, input_size);
			ReadMNIST<float>(filename_data, input_data, false);

			// Read input label [BUG FREE]
			Eigen::Tensor<float, 2> labels(data_size, 1);
			ReadMNIST<float>(filename_labels, labels, true);

			// Convert labels to 1 hot encoding [BUG FREE]
			Eigen::Tensor<float, 2> labels_encoded = OneHotEncoder<float, float>(labels, mnist_labels);

			if (is_training)
			{
				training_data = input_data;
				training_labels = labels_encoded;
			}
			else
			{
				validation_data = input_data;
				validation_labels = labels_encoded;
			}
		}

		void smoothLabels(const float& zero_offset, const float& one_offset) {
			training_labels = training_labels.unaryExpr(LabelSmoother<float>(zero_offset, one_offset));
			validation_labels = validation_labels.unaryExpr(LabelSmoother<float>(zero_offset, one_offset));
		};

		void unitScaleData() {
			training_data = training_data.unaryExpr(UnitScale<float>(training_data));
			validation_data = validation_data.unaryExpr(UnitScale<float>(validation_data));
		};

		void centerUnitScaleData() {
			training_data = training_data.unaryExpr(LinearScale<float>(0, 255, -1, 1));
			validation_data = validation_data.unaryExpr(LinearScale<float>(0, 255, -1, 1));
		};

		// Data attributes
		std::vector<float> mnist_labels = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

		// Data
		Eigen::Tensor<float, 2> training_data;
		Eigen::Tensor<float, 2> validation_data;
		Eigen::Tensor<float, 2> training_labels;
		Eigen::Tensor<float, 2> validation_labels;

		// Internal iterators
		int mnist_sample_start_training = 0;
		int mnist_sample_end_training = 0;
		int mnist_sample_start_validation = 0;
		int mnist_sample_end_validation = 0;
  };
}

#endif //SMARTPEAK_MNISTSIMULATOR_H