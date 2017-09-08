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
	vector<string> personNames;
	Size faceSize;

public:

	PersonRecognizer(int radius = 3, int neighbors = 8,
		int grid_x = 8, int grid_y = 8, double threshold = 90);
	~PersonRecognizer();

	void train(const string &facesListPath, vector<Mat> &faces_empty, vector<int> &labels_empty);
	bool recognize(const Mat &face, string &person, double &confidence) const;
	void load(const string &yml_file_path, const string &name_file_path);
	void save(const string &file_path) const;
	void readFacesList(const string &facesListPath, vector<Mat> &faces_empty, vector<int> &labels_empty, char seperator = ';');
	void readPersonNames(const string &namesFilePath);
	void matchLabel(const int &label, string &person) const;
};
#endif //PERSONRECOGNIZER_H