#ifndef PERSONRECOGNIZER_H
#define PERSONRECOGNIZER_H

#include <iostream>
#include <string>
#include <fstream>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\face.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;

class PersonRecognizer {

private:

	Ptr<FaceRecognizer> model;
	Size faceSize;
	string facesListPath;
	string dictionaryPath;

public:

	PersonRecognizer(vector<Mat> &faces_empty, vector<int> &labels_empty,
		const string &facesListPath, const string &dictionaryPath, int radius, int neighbors,
		int grid_x, int grid_y, double threshold);
	~PersonRecognizer();

	void train(vector<Mat> &faces_empty, vector<int> &labels_empty);
	bool recognize(const Mat &face, string &person, double &confidence) const;
	void readFacesList(vector<Mat> &faces_empty, vector<int> &labels_empty, char seperator = ';');
	void matchLabel(const int &label, string &person) const;
};
#endif //PERSONRECOGNIZER_H