#include "EyeDetector.h"

//EyeDetector::EyeDetector() : ObjectDetector(
//	"cascades/haarcascade_frontalface_default.xml",
//	1.02,
//	40,
//	Size(),
//	Size(300, 300)
//) {}

EyeDetector::EyeDetector(
	const string &cascadePath,
	double scaleFactor,
	int    minNeighbors,
	Size   minSize,
	Size   maxSize) : ObjectDetector(
		cascadePath,
		scaleFactor,
		minNeighbors,
		minSize,
		maxSize) {}

EyeDetector::~EyeDetector() {}


bool EyeDetector::detectEye(const Mat &image, Point &center) {

	return ObjectDetector::detectLargestObject(image, center);
}

bool EyeDetector::detectEye(const Mat &image, Rect &eye) {

	return ObjectDetector::detectLargestObject(image, eye);
}

bool EyeDetector::detectEye(const Mat &image, Rect &eye, Point &center) {

	return ObjectDetector::detectLargestObject(image, eye, center);
}