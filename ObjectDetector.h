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

	void detectObjects(const Mat &image, vector<Rect> &objects, int detectMode = 0);
	void detectLargestObject(const Mat &image, Rect &object);
	void detectLargestObject(const Mat &image, Rect &object, Point &center);
};
#endif //OBJECT_DETECTOR