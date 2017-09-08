#include "FaceAligner.h"

//FaceAligner::FaceAligner() {
//
//	desiredLeftEyeX = 0.16;
//	desiredLeftEyeY = 0.14;
//	desiredRightEyeX = (1.0f - desiredLeftEyeX);
//
//	faceSize = Size(150, 150);
//
//	led = new EyeDetector();
//	red = new EyeDetector();
//}

FaceAligner::FaceAligner(const double &desiredLeftEyeX,
	const double &desiredLeftEyeY, const Size &faceSize) :
	desiredLeftEyeX(desiredLeftEyeX), desiredLeftEyeY(desiredLeftEyeY),
	desiredRightEyeX(1.0f - desiredLeftEyeX), faceSize(faceSize) {}

void FaceAligner::align(Mat &face_image, Mat &warped_face_image, const Point &leftEye, const Point &rightEye) const {

	Mat face_image_gray = face_image.clone();

	if (face_image_gray.channels() == 3) {

		cvtColor(face_image_gray, face_image_gray, CV_BGR2GRAY);
	}
	else if (face_image_gray.channels() == 4) {

		cvtColor(face_image_gray, face_image_gray, CV_BGRA2GRAY);
	}

	Point2f eyesCenter = Point2f((leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f);

	double dx = (rightEye.x - leftEye.x);
	double dy = (rightEye.y - leftEye.y);

	double len = sqrt(dx*dx + dy*dy);

	double angle = atan2(dy, dx) * 180.0 / CV_PI;

	int desiredFaceWidth = faceSize.width;
	int desiredFaceHeight = faceSize.height;

	double desiredLen = (desiredRightEyeX - desiredLeftEyeX) * desiredFaceWidth;

	double scale = desiredLen / len;

	Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);

	rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
	rot_mat.at<double>(1, 2) += desiredFaceHeight * desiredLeftEyeY - eyesCenter.y;

	Mat warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128));
	warpAffine(face_image_gray, warped, rot_mat, warped.size());

	warped.copyTo(warped_face_image);
}