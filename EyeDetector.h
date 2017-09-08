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

	//EyeDetector();
	EyeDetector(
		const string &cascadePath,
		double scaleFactor = 1.02,
		int minNeighbors = 40,
		Size minSize = Size(),
		Size maxSize = Size(300, 300));
	~EyeDetector();

	bool detectEye(const Mat &image, Point &center);
	bool detectEye(const Mat &image, Rect &eye);
	bool detectEye(const Mat &image, Rect &eye, Point &center);
};
#endif //EYEDETECTOR_H