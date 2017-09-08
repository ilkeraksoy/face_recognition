#include "EyeDetector.h"

EyeDetector::EyeDetector() : ObjectDetector(
	"cascades/haarcascade_frontalface_default.xml",
	1.05,
	45,
	Size(2, 2),
	Size(50, 50)
) {}

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