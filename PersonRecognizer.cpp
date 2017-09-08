#include "PersonRecognizer.h"

PersonRecognizer::PersonRecognizer(int radius, int neighbors,
	int grid_x, int grid_y, double threshold) {

	//faceSize = Size(faces_empty[0].size().width, faces_empty[0].size().height);

	faceSize = Size(150, 150);

	model = LBPHFaceRecognizer::create(radius, neighbors, grid_x, grid_y, threshold);
}

PersonRecognizer::~PersonRecognizer() {}

void PersonRecognizer::train(const string &facesListPath, vector<Mat> &faces_empty, vector<int> &labels_empty) {

	readFacesList(facesListPath, faces_empty, labels_empty);

	model->train(faces_empty, labels_empty);
}


bool PersonRecognizer::recognize(const Mat &face, string &person, double &confidence) const {

	Mat face_gray = face.clone();

	if (face_gray.channels() == 3) {

		cvtColor(face_gray, face_gray, CV_BGR2GRAY);
	}
	else if (face_gray.channels() == 4) {

		cvtColor(face_gray, face_gray, CV_BGRA2GRAY);
	}

	if (face_gray.cols != faceSize.width || face_gray.rows != faceSize.height) {

		resize(face_gray, face_gray, faceSize, 1.0, 1.0, INTER_CUBIC);
	}

	int label;
	model->predict(face_gray, label, confidence);

	matchLabel(label, person);

	return label != -1 ? true : false;
}

void PersonRecognizer::load(const string &yml_file_path, const string &name_file_path) {

	model->read(yml_file_path);

	readPersonNames(name_file_path);
}

void PersonRecognizer::save(const string &file_path) const {

	model->write(file_path);
}

void PersonRecognizer::readFacesList(const string &facesListPath, vector<Mat> &faces_empty, vector<int> &labels_empty, char seperator) {

	ifstream facesListFile(facesListPath.c_str(), ios::in);

	if (!facesListFile) {

		string message_error = facesListPath + " No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, message_error);
	}

	string line, path, label, person;

	while (getline(facesListFile, line)) {

		stringstream lines(line);

		getline(lines, path, seperator);
		getline(lines, label, seperator);
		getline(lines, person);

		if (!path.empty() && !label.empty()) {

			faces_empty.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
			labels_empty.push_back(atoi(label.c_str()));

			cout << path << endl << label << endl;

			if (personNames.empty()) {

				personNames.push_back(person);
			}
			else if (person.compare(personNames[personNames.size() - 1])) {

				personNames.push_back(person);
			}
		}
	}

	facesListFile.close();
}

void PersonRecognizer::readPersonNames(const string &namesFilePath) {

	ifstream namesFile(namesFilePath.c_str(), ios::in);

	if (!namesFile) {

		string message_error = namesFilePath + " No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, message_error);
	}

	string name;

	while (getline(namesFile, name)) {

		if (personNames.empty()) {

			personNames.push_back(name);
		}
		else if (name.compare(personNames[personNames.size() - 1])) {

			personNames.push_back(name);
		}
	}

	namesFile.close();
}


void PersonRecognizer::matchLabel(const int &label, string &person) const {

	if (label != -1) {

		person = personNames[label];
	}
	else {

		person = "Unknown";
	}
}

//void PersonRecognizer::matchLabel(const int &label, string &person) const {
//
//	ifstream dictionaryFile(dictionaryPath.c_str(), ios::in);
//
//	if (!dictionaryFile) {
//
//		string message_error = dictionaryPath + " No valid input file was given, please check the given filename!..";
//		CV_Error(CV_StsBadArg, message_error);
//	}
//
//	string line, id, name;
//
//	while (getline(dictionaryFile, line)) {
//
//		stringstream lines(line);
//
//		getline(lines, id, ';');
//		getline(lines, name);
//
//		if (id.compare(to_string(label)) == 0) {
//
//			dictionaryFile.close();
//
//			person=name;
//
//			return;
//		}
//
//	}
//
//	dictionaryFile.close();
//
//	person = "Unknown Person";
//}