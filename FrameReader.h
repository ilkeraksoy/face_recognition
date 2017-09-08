#ifndef FRAMEREADER_H
#define FRAMEREADER_H

#include <iostream>
#include <string>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>


using namespace std;
using namespace cv;



class FrameReader {

private:

	VideoCapture capture;

	int endFrame,
		delta;

	const enum MODE { CAMERA_CAPTURE_MODE, VIDEO_CAPTURE_MODE };

	MODE CAPTURE_MODE;

public:

	FrameReader(const int &cameraId = 0);
	FrameReader(const string &videoPath, int startFrame = -1, int endFrame = -1, int delta = -1);
	~FrameReader();

	bool getNext(Mat &frame);
	void setFrameSize(int width, int height);
	Size getFrameSize();
};
#endif //FRAMEREADER_H