#include "ObjectDetector.h"

#include <vector>

ObjectDetector::ObjectDetector(
	const string &cascadePath,
	double scaleFactor,
	int    minNeighbors,
	Size   minSize,
	Size   maxSize) :
	scaleFactor(scaleFactor),
	minNeighbors(minNeighbors),
	minSize(minSize),
	maxSize(maxSize) {

	try {

		cascade.load(cascadePath);
	}
	catch (cv::Exception e) {}

	if (cascade.empty()) {

		cerr << "ERROR: Couldn't load Face Detector (";
		cerr << cascadePath << ")!" << endl;
		exit(1);
	}
}

ObjectDetector::~ObjectDetector() {}

void ObjectDetector::detectObjects(const Mat &image, vector<Rect> &objects, int detectMode) {

	Mat image_gray;

	if (image.channels() == 3) {

		cvtColor(image, image_gray, CV_BGR2GRAY);
	}
	else if (image.channels() == 4) {

		cvtColor(image, image_gray, CV_BGRA2GRAY);
	}
	else {

		image_gray = image;
	}

	equalizeHist(image_gray, image_gray);

	objects.clear();

	cascade.detectMultiScale(image_gray, objects, scaleFactor, minNeighbors, detectMode,
		minSize, maxSize);
}

bool ObjectDetector::detectLargestObject(const Mat &image, Point &center) {

	vector<Rect> objects;

	detectObjects(image, objects, CASCADE_FIND_BIGGEST_OBJECT);

	if (objects.size() > 0) {

		center = Point(objects[0].x + objects[0].width / 2, objects[0].y + objects[0].height / 2);

		return true;
	}
	else {

		center = Point(-1, -1);

		return false;
	}
}

void ObjectDetector::detectLargestObject(const Mat &image, Rect &object) {

	vector<Rect> objects;

	detectObjects(image, objects, CASCADE_FIND_BIGGEST_OBJECT);

	if (objects.size() > 0) {

		object = objects[0];
	}
	else {

		object = Rect(-1, -1, -1, -1);
	}
}

void ObjectDetector::detectLargestObject(const Mat &image, Rect &object, Point &center) {

	vector<Rect> objects;

	detectObjects(image, objects, CASCADE_FIND_BIGGEST_OBJECT);

	if (objects.size() > 0) {

		object = objects[0];
		center = Point(object.x + object.width / 2, object.y + object.height / 2);
	}
	else {

		object = Rect(-1, -1, -1, -1);
		center = Point(-1, -1);
	}
}