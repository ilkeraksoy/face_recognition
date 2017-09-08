#include "FaceDetector.h"
#include "FrameReader.h"
#include "FrameWriter.h"
#include "PersonRecognizer.h"
#include "Defines.h"

#include <iostream>
#include <fstream>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\objdetect.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {

	FrameReader fr(INPUT_VIDEO_PATH, START_FRAME, END_FRAME, FRAMES_DELTA);
	Size frameSize(fr.getFrameSize());

#ifdef WRITE_OUTPUT
	FrameWriter fw(OUTPUT_VIDEO_PATH, OUTPUT_VIDEO_FPS, frameSize, OUTPUT_VIDEO_FOURCC);
#endif

	FaceDetector fd(CASCADE_PATH, DETECT_SCALE_FACTOR, DETECT_MIN_NEIGHBORS, DETECT_MIN_SIZE, DETECT_MAX_SIZE);

	vector<Mat> faces;
	vector<int> labels;
	PersonRecognizer pr(faces, labels, FACES_LIST_PATH, DICTIONARY_PATH, LBPH_RADIUS, LBPH_RADIUS, LBPH_GRID_X, LBPH_GRID_Y, LBPH_THRESHOLD);

	vector<Rect> faces_r;
	Mat f;

	namedWindow(MAIN_WINDOW_NAME, WINDOW_AUTOSIZE | WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);
	namedWindow(MINI_WINDOW_NAME, WINDOW_AUTOSIZE | WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);

	int c = START_FRAME == -1 ? 0 : START_FRAME - 1;

	while (fr.getNext(f) && waitKey(20) != 27) {

		c++;

		bool has_match = false;
		double match_conf = 0;
		fd.detectFaces(f, faces_r);

		for (vector<Rect>::const_iterator face_r = faces_r.begin(); face_r != faces_r.end(); face_r++) {

			Scalar color = NO_MATCH_COLOR;
			Mat face_image = f(*face_r);

			cvtColor(face_image, face_image, CV_BGR2GRAY);
			equalizeHist(face_image, face_image);

			resize(face_image, face_image, FACE_SIZE, 1.0, 1.0, INTER_CUBIC);

			//int edge_size = max(face->width, face->height);

			//Rect square(face->x, face->y, edge_size, edge_size);	

			//Point center_ellipse(face->width * 0.5, face->height * 0.5);

			Point center_ellipse(75, 75);

			//Mat whiteImage(face->width, face->height, CV_8UC1, Scalar(0, 0, 0));

			Mat whiteImage(FACE_SIZE, CV_8UC1, Scalar(0, 0, 0));


			//cv::ellipse(whiteImage, center_ellipse, Size(face->width / 2 - 8, face->height / 2), 0, 0, 360, Scalar(255, 255, 255), -1, 8);

			cv::ellipse(whiteImage, center_ellipse, Size(75 - 8, 75), 0, 0, 360, Scalar(255, 255, 255), -1, 8);

			Mat res;
			bitwise_and(face_image, whiteImage, res);

			//resize(res, res, FACE_SIZE);	

#ifdef SHOW_DETECTED_FACE
			imshow(MINI_WINDOW_NAME, res);
#endif

			double confidence = 0;
			//bool face_match = false;
			int prediction;
			string personName;

			if (pr.recognize(res, personName, confidence)) {

				color = MATCH_COLOR;
				has_match = true;
				//face_match = true;
				match_conf = confidence;
			}




			Point center(face_r->x + face_r->width * 0.5, face_r->y + face_r->height * 0.5);
			circle(f, center, FACE_RADIUS_RATIO * face_r->width, color, CIRCLE_THICKNESS, LINE_TYPE, 0);

			Point text(face_r->x, face_r->y - face_r->height * 0.3);
			putText(f, personName, text, FONT_HERSHEY_PLAIN, 1.5, color, 1.5, CV_AA);
		}


		putText(f, "Face Recognition Demo", POS_TITLE,
			FONT, SCALE_TITLE, FONT_COLOR, THICKNESS_TITLE, LINE_TYPE);

		putText(f, format("Faces: %d", faces_r.size()), cvPoint(10, f.rows - 55),
			FONT, 2, FONT_COLOR, 1, LINE_TYPE);

		putText(f, format("Frame: %d", c), cvPoint(10, f.rows - 80),
			FONT, 2, FONT_COLOR, 1, LINE_TYPE);

		putText(f, format("Faces: %d", faces_r.size()), cvPoint(10, f.rows - 55),
			FONT, 2, FONT_COLOR, 1, LINE_TYPE);

		putText(f, format("Match: %s", has_match ? "True" : "False"), cvPoint(10, f.rows - 30),
			FONT, 2, FONT_COLOR, 1, LINE_TYPE);

		putText(f, format("Confidence: %f", has_match ? match_conf : 0), cvPoint(10, f.rows - 5),
			FONT, 2, FONT_COLOR, 1, LINE_TYPE);

#ifdef WRITE_OUTPUT
		FrameWriter fw(OUTPUT_VIDEO_PATH, OUTPUT_VIDEO_FPS, frameSize, OUTPUT_VIDEO_FOURCC);
#endif

#ifdef SHOW_OUTPUT
		imshow(MAIN_WINDOW_NAME, f);
#endif

	}

	exit(0);
}