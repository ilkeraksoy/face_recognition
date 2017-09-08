#include "EyeDetector.h"
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

	FrameReader fr(0);
	Size frameSize(fr.getFrameSize());

#ifdef WRITE_OUTPUT
	FrameWriter fw(OUTPUT_VIDEO_PATH, OUTPUT_VIDEO_FPS, frameSize, OUTPUT_VIDEO_FOURCC);
#endif

	FaceDetector fd(FACE_CASCADE_PATH, FACE_DETECT_SCALE_FACTOR, FACE_DETECT_MIN_NEIGHBORS, FACE_DETECT_MIN_SIZE, FACE_DETECT_MAX_SIZE);
	EyeDetector le(LEFT_EYE_CASCADE_PATH, EYE_DETECT_SCALE_FACTOR, EYE_DETECT_MIN_NEIGHBORS, EYE_DETECT_MIN_SIZE, EYE_DETECT_MAX_SIZE);
	EyeDetector re(RIGHT_EYE_CASCADE_PATH, EYE_DETECT_SCALE_FACTOR, EYE_DETECT_MIN_NEIGHBORS, EYE_DETECT_MIN_SIZE, EYE_DETECT_MAX_SIZE);


	vector<Mat> faces;
	vector<int> labels;
	PersonRecognizer pr(faces, labels, FACES_LIST_PATH, DICTIONARY_PATH, LBPH_RADIUS, LBPH_NEIGHBORS, LBPH_GRID_X, LBPH_GRID_Y, LBPH_THRESHOLD);

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

			Mat face_image = f(*face_r);


			int leftX = cvRound(face_image.cols * EYE_SX);
			int topY = cvRound(face_image.rows * EYE_SY);
			int widthX = cvRound(face_image.cols * EYE_SW);
			int heightY = cvRound(face_image.rows * EYE_SH);
			int rightX = cvRound(face_image.cols * (1.0 - EYE_SX - EYE_SW));
			Mat topLeftOfFace = face_image(Rect(leftX, topY, widthX,
				heightY));
			Mat topRightOfFace = face_image(Rect(rightX, topY, widthX,
				heightY));

			Point leftEye, rightEye;

			if (le.detectEye(topLeftOfFace, leftEye) && re.detectEye(topRightOfFace, rightEye)) {

				circle(topLeftOfFace, leftEye, 2, Scalar(255, 0, 0), 1, LINE_TYPE, 0);
				circle(topRightOfFace, rightEye, 2, Scalar(255, 0, 0), 1, LINE_TYPE, 0);

				cvtColor(face_image, face_image, CV_BGR2GRAY);
				equalizeHist(face_image, face_image);

				resize(face_image, face_image, FACE_SIZE, 1.0, 1.0, INTER_CUBIC);

				Point center_ellipse(75, 75);

				Mat whiteImage(FACE_SIZE, CV_8UC1, Scalar(0, 0, 0));

				cv::ellipse(whiteImage, center_ellipse, Size(75 - 8, 75), 0, 0, 360, Scalar(255, 255, 255), -1, 8);

				Mat res;
				bitwise_and(face_image, whiteImage, res);

#ifdef SHOW_DETECTED_FACE
				imshow(MINI_WINDOW_NAME, res);
#endif

				double confidence = 0;
				bool face_match = false;
				int prediction;
				string personName;

				Scalar color = NO_MATCH_COLOR;

				if (pr.recognize(res, personName, confidence)) {

					color = MATCH_COLOR;
					has_match = true;
					face_match = true;
					match_conf = confidence;
				}




				Point center(face_r->x + face_r->width * 0.5, face_r->y + face_r->height * 0.5);
				circle(f, center, FACE_RADIUS_RATIO * face_r->width, color, CIRCLE_THICKNESS, LINE_TYPE, 0);

				//Point text(face_r->x + face_r->width / 4, face_r->y - 2);
				//putText(f, personName, text, FONT_HERSHEY_PLAIN, 1.5, color, 1.5, CV_AA);
			}
		}


		putText(f, "Face Recognition Demo", POS_TITLE,
			FONT, SCALE_TITLE, FONT_COLOR, THICKNESS_TITLE, LINE_TYPE);

		putText(f, format("Faces: %d", faces_r.size()), cvPoint(10, f.rows - 55),
			FONT, 2, FONT_COLOR, 1, LINE_TYPE);

		putText(f, format("Frame: %d", c), cvPoint(10, f.rows - 80),
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

	destroyAllWindows();
	return 0;
}