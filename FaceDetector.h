#ifndef FACEDETECTOR_H
#define	FACEDETECTOR_H

#include "ObjectDetector.h"
#include "EyeDetector.h"

#include <iostream>
#include <string>

#include <opencv2\core.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;

const double EYE_SX = 0.10;
const double EYE_SY = 0.19;
const double EYE_SW = 0.40;
const double EYE_SH = 0.36;

class FaceDetector :public ObjectDetector {

public:

	EyeDetector *led, *red;

	const enum MODE { WITHOUT_EYE_DETECTOR_MODE, WITH_EYE_DETECTOR_MODE };

	MODE DETECT_MODE;

	//FaceDetector();
	FaceDetector(
		const string &cascadePath,
		double	     scaleFactor = 1.02,
		int          minNeighbors = 40,
		Size         minSize = Size(),
		Size         maxSize = Size());
	FaceDetector(
		const string &faceCascadePath,
		const string &leftEyeCascadePath,
		const string &rightEyeCascadePath,
		double	     scaleFactor = 1.02,
		int          minNeighbors = 40,
		Size         minSize = Size(),
		Size         maxSize = Size());
	~FaceDetector();

	bool detectFaces(const Mat &image, vector<Rect> &faces, int cascadeMode = 0);
	bool detectFaces(const Mat &image, vector<Rect> &faces, vector<Rect> &leftEyes, vector<Rect> &rightEyes, int cascadeMode = 0);
	bool detectLargestFace(const Mat &image, Rect &face);
	bool detectLargestFace(const Mat &image, Rect &face, Point &center);
};
#endif //FACEDETECTOR_H