#include "FaceDetector.h"

#include <vector>

FaceDetector::FaceDetector() : ObjectDetector(
	"cascades/haarcascade_frontalface_default.xml",
	1.05,
	45,
	Size(),
	Size()
) {}

FaceDetector::FaceDetector(
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

FaceDetector::~FaceDetector() {}



void FaceDetector::detectFaces(const Mat &image, vector<Rect> &objects, int detectMode) {

	ObjectDetector::detectObjects(image, objects, detectMode);
}

void FaceDetector::detectLargestFace(const Mat &image, Rect &object) {

	ObjectDetector::detectLargestObject(image, object);
}

void FaceDetector::detectLargestFace(const Mat &image, Rect &object, Point &center) {

	ObjectDetector::detectLargestObject(image, object, center);
}