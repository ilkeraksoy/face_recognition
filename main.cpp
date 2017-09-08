#include <iostream>
#include <fstream>
#include <sstream>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/face.hpp"


using namespace std;
using namespace cv;
using namespace cv::face;

void read_csv(string &fileName, vector<Mat> &images, vector<int> &labels, char seperator);
string matchId(string &fileName, int &label_id);
void readFromVideoFile(int, string, string, string, string, string);


int main(int argc, char *argv[]) {

	readFromVideoFile(argc, argv[0], argv[1], argv[2], argv[3], argv[4]);

	system("PAUSE");

	return 0;
}



//read csv file
void read_csv(string &fileName, vector<Mat> &images, vector<int> &labels, char seperator = ';') {

	ifstream file(fileName.c_str(), ios::in);

	if (!file) {

		string message_error = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, message_error);


	}

	string line, path, classLabel;


	while (getline(file, line)) {

		stringstream lines(line);

		getline(lines, path, seperator);
		getline(lines, classLabel);

		if (!path.empty() && !classLabel.empty()) {

			Mat image = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
			//equalizeHist(image,image);

			images.push_back(image);
			labels.push_back(atoi(classLabel.c_str()));
		}
	}

	file.close();
}


//match returned label
string matchId(string &fileName, int &label_id) {

	ifstream file(fileName.c_str(), ios::in);

	string line, id, name;

	while (getline(file, line)) {

		stringstream lines(line);

		getline(lines, id, ';');
		getline(lines, name);

		if (id.compare(to_string(label_id)) == 0) {

			file.close();
			return name;
		}

	}

	return "Unknown Person";
}

void readFromVideoFile(int argc, string argv0, string argv1, string argv2, string argv3, string argv4) {

	if (argc != 5) {

		cout << "Usage: " << argv0 << " <haarcascade_file> <csv_file> <dictionary_file> <video_file>" << endl;
		exit(1);
	}



	//    string fileName_haarcascade=string(argv[1]);
	//    string fileName_csv=string(argv[2]);
	//    int deviceID=atoi(argv[3]);

	//required files directories
	string fileName_haarcascade = argv1;
	string fileName_csv = argv2;
	string fileName_dictionary = argv3;
	string fileName_video = argv4;

	//empty vectors for faces and labels
	vector<Mat> images_input;
	vector<int> labels;

	try {

		read_csv(fileName_csv, images_input, labels, ';');
	}
	catch (Exception &e) {

		cerr << "Error opening file: " << fileName_csv << ". Reason: " << e.msg << endl;
		exit(1);
	}

	int width_images = images_input[0].cols;
	int height_images = images_input[0].rows;

	//training
	Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create(1, 16, 8, 8, 160.0);
	model->train(images_input, labels);

	//haar cascade file
	CascadeClassifier frontFace;
	frontFace.load(fileName_haarcascade);

	//video capture class instance
	VideoCapture cap;
	cap.open(fileName_video);

	if (!cap.isOpened()) {

		cerr << "Connection failed..!" << endl;

		exit(1);
	}
	else {

		cout << "Connection successful..." << endl;

		//frame from camera
		Mat frame, frame_gray;

		//create window
		namedWindow("FaceRecognition", WINDOW_AUTOSIZE | WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);
		namedWindow("DetectedFace", WINDOW_AUTOSIZE | WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);

		//returned id
		int prediction = -1;

		string predictionName;

		//until press ESC
		while (waitKey(20) != 27) {

			//assign frame
			cap >> frame;


			//convert grayscale
			cvtColor(frame, frame_gray, CV_BGR2GRAY);

			//histogram equalization
			equalizeHist(frame_gray, frame_gray);

			//found faces
			vector<Rect_<int>> faces;

			//face detection
			frontFace.detectMultiScale(frame_gray, faces, 1.1, 30, cv::CASCADE_SCALE_IMAGE, Size(50, 50));

			//for each found face...
			for (int i = 0; i<faces.size(); i++) {

				//face rectangle
				Rect face_i = faces[i];

				//copy face boundary submatrix
				Mat face = frame_gray(face_i);

				//show it
				imshow("DetectedFace", face);

				//scaled face
				Mat face_resized;

				//scaling
				resize(face, face_resized, Size(width_images, height_images), 1.0, 1.0, INTER_CUBIC);

				//query face
				prediction = model->predict(face_resized);

				//match id for person name
				predictionName = matchId(fileName_dictionary, prediction);

				string text_rectangle = predictionName;

				//draw ellipse around face
				Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);

				ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2),
					0, 0, 360, CV_RGB(204, 0, 102), 4, 8, 0);


				int position_x = max(face_i.tl().x, 0);

				int position_y = max(face_i.tl().y - 2, 0);

				//draw text
				putText(frame, text_rectangle, Point(position_x, position_y), FONT_HERSHEY_PLAIN, 1.5, CV_RGB(0, 255, 0), 1.5, CV_AA);

			}

			//show frame
			imshow("FaceRecognition", frame);

			//char key = (char) waitKey(20);

			//if(key==27)
			//    break;
		}

		cap.release();
		exit(EXIT_SUCCESS);
	}
}

