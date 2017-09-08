#ifndef FACEDETECTOR_H
#define	FACEDETECTOR_H

#include "ObjectDetector.h"

#include <iostream>
#include <string>

#include <opencv2\core.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;

class FaceDetector :public ObjectDetector {

public:

	FaceDetector();
	FaceDetector(
		const string &cascadePath,
		double	     scaleFactor = 1.05,
		int          minNeighbors = 40,
		Size         minSize = Size(10, 10),
		Size         maxSize = Size());
	~FaceDetector();

	void detectFaces(const Mat &image, vector<Rect> &objects, int detectMode = 0);
	void detectLargestFace(const Mat &image, Rect &object);
	void detectLargestFace(const Mat &image, Rect &object, Point &center);
};
#endif //FACEDETECTOR_H