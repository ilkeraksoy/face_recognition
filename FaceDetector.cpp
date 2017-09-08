#include "FaceDetector.h"

#include <vector>

//FaceDetector::FaceDetector() : ObjectDetector(
//	"cascades/haarcascade_frontalface_default.xml",
//	1.05,
//	45,
//	Size(),
//	Size()
//) {
//
//	led = new EyeDetector();
//	red = new EyeDetector();
//}

FaceDetector::FaceDetector(
	const string &faceCascadePath,
	double scaleFactor,
	int    minNeighbors,
	Size   minSize,
	Size   maxSize) : ObjectDetector(
		faceCascadePath,
		scaleFactor,
		minNeighbors,
		minSize,
		maxSize) {

	DETECT_MODE = WITHOUT_EYE_DETECTOR_MODE;
}

FaceDetector::FaceDetector(
	const string &faceCascadePath,
	const string &leftEyeCascadePath,
	const string &rightEyeCascadePath,
	double scaleFactor,
	int    minNeighbors,
	Size   minSize,
	Size   maxSize) : ObjectDetector(
		faceCascadePath,
		scaleFactor,
		minNeighbors,
		minSize,
		maxSize) {

	DETECT_MODE = WITH_EYE_DETECTOR_MODE;

	led = new EyeDetector(leftEyeCascadePath);
	red = new EyeDetector(rightEyeCascadePath);
}

FaceDetector::~FaceDetector() {}

bool FaceDetector::detectFaces(const Mat &image, vector<Rect> &faces, int cascadeMode) {

	if (DETECT_MODE == WITH_EYE_DETECTOR_MODE) {

		faces.clear();

		vector<Rect> faces_r;

		if (ObjectDetector::detectObjects(image, faces_r, cascadeMode)) {

			for (vector<Rect>::const_iterator face_r = faces_r.begin(); face_r != faces_r.end(); face_r++) {

				Mat face_image = image(*face_r);

				int leftX = cvRound(face_image.cols * EYE_SX);
				int topY = cvRound(face_image.rows * EYE_SY);
				int widthX = cvRound(face_image.cols * EYE_SW);
				int heightY = cvRound(face_image.rows * EYE_SH);
				int rightX = cvRound(face_image.cols * (1.0 - EYE_SX - EYE_SW));

				Mat topLeftOfFace = face_image(Rect(leftX, topY, widthX,
					heightY));
				Mat topRightOfFace = face_image(Rect(rightX, topY, widthX,
					heightY));

				Rect leftEye, rightEye;

				if (led->detectEye(topLeftOfFace, leftEye) && red->detectEye(topRightOfFace, rightEye)) {

					faces.push_back(*face_r);
				}
			}

			if (faces.size() > 0) {

				return true;
			}
			else {

				return false;
			}
		}
		else {

			return false;
		}
	}
	else if (DETECT_MODE == WITHOUT_EYE_DETECTOR_MODE) {

		faces.clear();

		vector<Rect> faces_r;

		return ObjectDetector::detectObjects(image, faces, cascadeMode);
	}
	else {

		return false;
	}
}

bool FaceDetector::detectFaces(const Mat &image, vector<Rect> &faces, vector<Rect> &leftEyes, vector<Rect> &rightEyes, int cascadeMode) {

	if (DETECT_MODE == WITH_EYE_DETECTOR_MODE) {

		faces.clear();
		leftEyes.clear();
		rightEyes.clear();

		vector<Rect> faces_r;

		if (ObjectDetector::detectObjects(image, faces_r, cascadeMode)) {

			for (vector<Rect>::const_iterator face_r = faces_r.begin(); face_r != faces_r.end(); face_r++) {

				Mat face_image = image(*face_r);

				int leftX = cvRound(face_image.cols * EYE_SX);
				int topY = cvRound(face_image.rows * EYE_SY);
				int widthX = cvRound(face_image.cols * EYE_SW);
				int heightY = cvRound(face_image.rows * EYE_SH);
				int rightX = cvRound(face_image.cols * (1.0 - EYE_SX - EYE_SW));

				Mat topLeftOfFace = face_image(Rect(leftX, topY, widthX,
					heightY));
				Mat topRightOfFace = face_image(Rect(rightX, topY, widthX,
					heightY));

				Rect leftEye, rightEye;

				if (led->detectEye(topLeftOfFace, leftEye) && red->detectEye(topRightOfFace, rightEye)) {

					faces.push_back(*face_r);

					leftEyes.push_back(Rect((*face_r).x + leftX + leftEye.x, (*face_r).y + topY + leftEye.y, leftEye.width, leftEye.height));
					rightEyes.push_back(Rect((*face_r).x + rightX + rightEye.x, (*face_r).y + topY + rightEye.y, rightEye.width, rightEye.height));
				}
			}

			if (faces.size() > 0) {

				return true;
			}
			else {

				return false;
			}
		}
		else {

			return false;
		}
	}
	else if (DETECT_MODE == WITHOUT_EYE_DETECTOR_MODE) {

		cout << "Detect Mode: WITHOUT_EYE_DETECTOR_MODE" << endl << "Should be WITH_EYE_DETECTOR_MODE" << endl << "Returning false..!" << endl;

		return false;
	}
	else {

		return false;
	}
}

bool FaceDetector::detectLargestFace(const Mat &image, Rect &face) {

	return ObjectDetector::detectLargestObject(image, face);
}

bool FaceDetector::detectLargestFace(const Mat &image, Rect &face, Point &center) {

	return ObjectDetector::detectLargestObject(image, face, center);
}