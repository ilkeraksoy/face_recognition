#ifndef FRAMEWRITER_H
#define	FRAMEWRITER_H

#include <string>

#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

class FrameWriter {

private:

	VideoWriter videoWriter;
	Size frameSize;

public:

	FrameWriter(const string videoPath, double fps, Size size, int fourcc);
	~FrameWriter();

	void write(Mat &frame);
};

#endif //FRAMEWRITER_H