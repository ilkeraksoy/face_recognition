#ifndef OBJECT_DETECTOR_H
#define	OBJECT_DETECTOR_H

#include <iostream>
#include <string>

#include <opencv2\core.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;

class ObjectDetector {

protected:

	CascadeClassifier cascade;

	double scaleFactor;
	int    minNeighbors;
	Size   minSize;
	Size   maxSize;

public:

	ObjectDetector(
		const string &cascadePath,
		double	     scaleFactor,
		int          minNeighbors,
		Size         minSize,
		Size         maxSize);
	virtual ~ObjectDetector();

	bool detectObjects(const Mat &image, vector<Rect> &objects, int cascadeMode);
	bool detectLargestObject(const Mat &image, Point &center);
	bool detectLargestObject(const Mat &image, Rect &object);
	bool detectLargestObject(const Mat &image, Rect &object, Point &center);
};
#endif //OBJECT_DETECTOR