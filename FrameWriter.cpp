#include "FrameWriter.h"

FrameWriter::FrameWriter(const string videoPath, double fps, Size frameSize, int fourcc) : frameSize(frameSize) {

	videoWriter.open(
		videoPath,
		fourcc,
		fps,
		frameSize,
		true
	);
}

void FrameWriter::write(Mat& frame) {
	videoWriter << frame;
}

FrameWriter::~FrameWriter() {
	videoWriter.release();
}