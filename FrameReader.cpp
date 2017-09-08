#include "FrameReader.h"

FrameReader::FrameReader(const int &cameraId) {

	CAPTURE_MODE = CAMERA_CAPTURE_MODE;

	capture.open(cameraId);

	if (!capture.isOpened()) {

		cout << "Connection failed!.." << endl;

		exit(EXIT_FAILURE);
	}
}


FrameReader::FrameReader(const string &videoPath, int startFrame, int endFrame, int delta) :
	endFrame(endFrame), delta(delta) {

	CAPTURE_MODE = VIDEO_CAPTURE_MODE;

	capture.open(videoPath);

	if (!capture.isOpened()) {

		cout << "Connection failed!.." << endl;

		exit(EXIT_FAILURE);
	}

	if (startFrame != -1 && startFrame > -1)
		capture.set(CV_CAP_PROP_POS_FRAMES, startFrame);
}

FrameReader::~FrameReader() {

	capture.release();
}

bool FrameReader::getNext(Mat &frame) {

	if (CAPTURE_MODE == VIDEO_CAPTURE_MODE) {

		if (delta != -1 && delta > 0)
			capture.set(CV_CAP_PROP_POS_FRAMES, capture.get(CV_CAP_PROP_POS_FRAMES) + delta);

		if (endFrame != -1 && endFrame > 0 && capture.get(CV_CAP_PROP_POS_FRAMES) == endFrame)
			return false;
	}

	return capture.read(frame);
}

void FrameReader::setFrameSize(int width, int height) {

	capture.set(CV_CAP_PROP_FRAME_WIDTH, width);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, height);
}

Size FrameReader::getFrameSize() {

	return Size(
		capture.get(CV_CAP_PROP_FRAME_WIDTH),
		capture.get(CV_CAP_PROP_FRAME_HEIGHT)
	);
}