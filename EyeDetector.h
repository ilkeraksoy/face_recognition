#ifndef EYEDETECTOR_H
#define EYEDETECTOR_H

#include "ObjectDetector.h"

#include <iostream>
#include <string>

#include <opencv2\core.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;

class EyeDetector :public ObjectDetector {

public:

	EyeDetector();
	EyeDetector(
		const string &cascadePath,
		double scaleFactor = 1.05,
		int minNeighbors = 40,
		Size minSize = Size(2, 2),
		Size maxSize = Size(50, 50));
	~EyeDetector();

	bool detectEye(const Mat &image, Point &center);
};
#endif //EYEDETECTOR_H