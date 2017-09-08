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

bool ObjectDetector::detectObjects(const Mat &image, vector<Rect> &objects, int cascadeMode) {

	Mat image_gray = image.clone();

	objects.clear();

	if (image_gray.channels() == 3) {

		cvtColor(image_gray, image_gray, CV_BGR2GRAY);
	}
	else if (image_gray.channels() == 4) {

		cvtColor(image_gray, image_gray, CV_BGRA2GRAY);
	}

	equalizeHist(image_gray, image_gray);

	cascade.detectMultiScale(image_gray, objects, scaleFactor, minNeighbors, cascadeMode,
		minSize, maxSize);

	if (objects.size() > 0) {

		return true;
	}
	else {

		return false;
	}
}

bool ObjectDetector::detectLargestObject(const Mat &image, Point &center) {

	Mat image_gray = image.clone();

	if (image_gray.channels() == 3) {

		cvtColor(image_gray, image_gray, CV_BGR2GRAY);
	}
	else if (image_gray.channels() == 4) {

		cvtColor(image_gray, image_gray, CV_BGRA2GRAY);
	}

	equalizeHist(image_gray, image_gray);

	vector<Rect> objects;

	detectObjects(image_gray, objects, CASCADE_FIND_BIGGEST_OBJECT);

	if (objects.size() > 0) {

		center = Point(objects[0].x + objects[0].width / 2, objects[0].y + objects[0].height / 2);

		return true;
	}
	else {

		center = Point(-1, -1);

		return false;
	}
}

bool ObjectDetector::detectLargestObject(const Mat &image, Rect &object) {

	Mat image_gray = image.clone();

	if (image_gray.channels() == 3) {

		cvtColor(image_gray, image_gray, CV_BGR2GRAY);
	}
	else if (image_gray.channels() == 4) {

		cvtColor(image_gray, image_gray, CV_BGRA2GRAY);
	}

	equalizeHist(image_gray, image_gray);

	vector<Rect> objects;

	detectObjects(image_gray, objects, CASCADE_FIND_BIGGEST_OBJECT);

	if (objects.size() > 0) {

		object = objects[0];

		return true;
	}
	else {

		object = Rect(-1, -1, -1, -1);

		return false;
	}
}

bool ObjectDetector::detectLargestObject(const Mat &image, Rect &object, Point &center) {

	Mat image_gray = image.clone();

	if (image_gray.channels() == 3) {

		cvtColor(image_gray, image_gray, CV_BGR2GRAY);
	}
	else if (image_gray.channels() == 4) {

		cvtColor(image_gray, image_gray, CV_BGRA2GRAY);
	}

	equalizeHist(image_gray, image_gray);

	vector<Rect> objects;

	detectObjects(image_gray, objects, CASCADE_FIND_BIGGEST_OBJECT);

	if (objects.size() > 0) {

		object = objects[0];
		center = Point(object.x + object.width / 2, object.y + object.height / 2);

		return true;
	}
	else {

		object = Rect(-1, -1, -1, -1);
		center = Point(-1, -1);

		return false;
	}
}