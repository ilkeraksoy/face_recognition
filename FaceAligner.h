#ifndef FACEALIGNER_H
#define FACEALIGNER_H

#include <iostream>
#include <string>

#include "Defines.h"
#include "EyeDetector.h"


#include <opencv2\core.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;

class FaceAligner {

private:

	EyeDetector *led, *red;

	double desiredLeftEyeX;
	double desiredLeftEyeY;
	double desiredRightEyeX;

	Size faceSize;

public:

	//FaceAligner();
	FaceAligner(const double &desiredLeftEyeX = 0.16, const double &desiredLeftEyeY = 0.14, const Size &faceSize = Size(150, 150));

	void align(Mat &face_image, Mat &warped_face_image, const Point &leftEye, const Point &rightEye) const;
};
#endif //FACEALIGNER_H