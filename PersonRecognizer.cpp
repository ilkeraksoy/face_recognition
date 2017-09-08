#include "PersonRecognizer.h"

PersonRecognizer::PersonRecognizer(vector<Mat> &faces_empty, vector<int> &labels_empty,
	const string &facesListPath,
	const string &dictionaryPath,
	int radius, int neighbors,
	int grid_x, int grid_y, double threshold) {

	this->facesListPath = facesListPath;
	this->dictionaryPath = dictionaryPath;

	//faceSize = Size(faces_empty[0].size().width, faces_empty[0].size().height);

	faceSize = Size(150, 150);

	model = LBPHFaceRecognizer::create(radius = 1, neighbors = 8, grid_x = 8, grid_y = 8, threshold = 120);

	train(faces_empty, labels_empty);
}

PersonRecognizer::~PersonRecognizer() {}

void PersonRecognizer::train(vector<Mat> &faces_empty, vector<int> &labels_empty) {

	readFacesList(faces_empty, labels_empty);

	model->train(faces_empty, labels_empty);
}


bool PersonRecognizer::recognize(const Mat &face, string &person, double &confidence) const {

	Mat face_gray;

	if (face.channels() == 3) {

		cvtColor(face, face_gray, CV_BGR2GRAY);
	}
	else if (face.channels() == 4) {

		cvtColor(face, face_gray, CV_BGRA2GRAY);
	}
	else {

		face_gray = face;
	}

	equalizeHist(face_gray, face_gray);

	resize(face_gray, face_gray, faceSize, 1.0, 1.0, INTER_CUBIC);

	int label;
	model->predict(face, label, confidence);

	matchLabel(label, person);

	return label != -1 ? true : false;
}

void PersonRecognizer::readFacesList(vector<Mat> &faces_empty, vector<int> &labels_empty, char seperator) {

	ifstream facesListFile(facesListPath.c_str(), ios::in);

	if (!facesListFile) {

		string message_error = facesListPath + " No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, message_error);
	}

	string line, path, label;

	while (getline(facesListFile, line)) {

		stringstream lines(line);

		getline(lines, path, seperator);
		getline(lines, label);

		if (!path.empty() && !label.empty()) {

			faces_empty.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
			labels_empty.push_back(atoi(label.c_str()));
		}
	}

	facesListFile.close();
}

void PersonRecognizer::matchLabel(const int &label, string &person) const {

	ifstream dictionaryFile(dictionaryPath.c_str(), ios::in);

	if (!dictionaryFile) {

		string message_error = dictionaryPath + " No valid input file was given, please check the given filename!..";
		CV_Error(CV_StsBadArg, message_error);
	}

	string line, id, name;

	while (getline(dictionaryFile, line)) {

		stringstream lines(line);

		getline(lines, id, ';');
		getline(lines, name);

		if (id.compare(to_string(label)) == 0) {

			dictionaryFile.close();

			person = name;

			return;
		}

	}

	dictionaryFile.close();

	person = "Unknown Person";
}