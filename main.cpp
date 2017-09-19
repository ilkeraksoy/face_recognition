#include "Include.h"

int main(int argc, char **argv) {

	//frame reader (0 is default camera.)	
	FrameReader fr(0);	
	
	//get frame size (width and height) from camera 
	Size frameSize(fr.getFrameSize());

	//frame writer (to disk)
#ifdef WRITE_OUTPUT
	FrameWriter fw(OUTPUT_VIDEO_PATH, OUTPUT_VIDEO_FPS, frameSize, OUTPUT_VIDEO_FOURCC);
#endif

	//face detector object	
	FaceDetector fd(FACE_CASCADE_PATH, LEFT_EYE_CASCADE_PATH, RIGHT_EYE_CASCADE_PATH, FACE_DETECT_SCALE_FACTOR, FACE_DETECT_MIN_NEIGHBORS, FACE_DETECT_MIN_SIZE, FACE_DETECT_MAX_SIZE);
	
	//face aligner object
	FaceAligner fa(FACE_ALIGN_DESIRED_LEFT_EYE_X, FACE_ALIGN_DESIRED_LEFT_EYE_Y, FACE_ALIGN_FACE_SIZE);

	//empty vectors for face images, label numbers and person names
	vector<Mat> faces;
	vector<int> labels;
	vector<string> personNames;

	//person recognizer object	
	PersonRecognizer pr(LBPH_RADIUS, LBPH_NEIGHBORS, LBPH_GRID_X, LBPH_GRID_Y, LBPH_THRESHOLD);

	//train	person recognizer with dataset
	pr.train(FACES_LIST_PATH, faces, labels);

	//save trained file
	//pr.save("faces/persons.yml");

	//load trained file
	//pr.load(LBPH_YML_FILE_PATH, LBPH_NAME_FILE_PATH);

	//get person names
	pr.getPersonNames(personNames);

	//prepare the main window	
	namedWindow(MAIN_WINDOW_NAME, WINDOW_AUTOSIZE | WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);

	//prepare detected face window	
#ifdef SHOW_DETECTED_FACE
	namedWindow(MINI_WINDOW_NAME, WINDOW_AUTOSIZE | WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);
#endif

	//frame counter	
	int c = START_FRAME == -1 ? 0 : START_FRAME - 1;

	//empty vectors for face rectangles, left eye rectangles and right eye rectangles	
	vector<Rect> faces_r;
	vector<Rect> leftEyes_r;
	vector<Rect> rightEyes_r;

	//frame from the camera and its copy
	Mat frame, frame_copy;

	//if the frame reads succeed and ESC key is not pressed	
	while (fr.getNext(frame) && waitKey(20) != 27) {
		
		//increase frame counter
		c++;

		//copy original frame		
		frame.copyTo(frame_copy);

		//match state, confidence and face counter		
		bool has_match = false;
		double match_conf = 0;
		int numberOfFaces = 0;

		//if face(s) detected and also its left eye(s) and right eye(s) detected		
		if (fd.detectFaces(frame_copy, faces_r, leftEyes_r, rightEyes_r)) {
			
			//until number of faces...
			for (vector<Rect>::const_iterator face_r = faces_r.begin(), eyeL_r = leftEyes_r.begin(), eyeR_r = rightEyes_r.begin();
				face_r != faces_r.end(); face_r++, eyeL_r++, eyeR_r++) {
				
				//cut face(s) from frame
				Mat face_image = frame_copy(*face_r).clone();

				//increase face counter
				numberOfFaces++;

				//convert to grayscale image				
				cvtColor(face_image, face_image, CV_BGR2GRAY);

				//left and right eye coordinates on frame				
				Point leftEyeOnFrame((*eyeL_r).x + (*eyeL_r).width / 2, (*eyeL_r).y + (*eyeL_r).height / 2);
				Point rightEyeOnFrame((*eyeR_r).x + (*eyeR_r).width / 2, (*eyeR_r).y + (*eyeR_r).height / 2);

				//draw circle around eyes				
				circle(frame, Point((*eyeL_r).x + (*eyeL_r).width / 2, (*eyeL_r).y + (*eyeL_r).height / 2), 2, Scalar(0, 0, 255), 2, CV_AA);
				circle(frame, Point((*eyeR_r).x + (*eyeR_r).width / 2, (*eyeR_r).y + (*eyeR_r).height / 2), 2, Scalar(0, 0, 255), 2, CV_AA);

				//left and right eye coordinates on face image
				Point leftEye((*eyeL_r).x - (*face_r).x + (*eyeL_r).width / 2, (*eyeL_r).y - (*face_r).y + (*eyeL_r).height / 2);
				Point rightEye((*eyeR_r).x - (*face_r).x + (*eyeR_r).width / 2, (*eyeR_r).y - (*face_r).y + (*eyeR_r).height / 2);

				//circle(face_image, leftEye, 2, Scalar(0, 0, 255), 2, CV_AA);
				//circle(face_image, rightEye, 2, Scalar(0, 0, 255), 2, CV_AA);
				
				//aligned face
				Mat aligned_face;

				//align face				
				fa.align(face_image, aligned_face, leftEye, rightEye);

				//left and right face				
				Mat leftFace, rightFace;

				//left and right face rectangles coordinates and its sizes
				int leftX, leftY, rightX, rightY;
				int leftWidth, leftHeight, rightWidth, rightHeight;

				leftX = 0;
				leftY = 0;

				rightX = aligned_face.cols / 2;
				rightY = 0;

				leftWidth = aligned_face.cols / 2;
				leftHeight = aligned_face.rows;

				rightWidth = leftWidth;
				rightHeight = leftHeight;

				//cut left and right face from aligned face
				leftFace = aligned_face(Rect(leftX, leftY, leftWidth, leftHeight));
				rightFace = aligned_face(Rect(rightX, rightY, rightWidth, rightHeight));

				//histogram equalization operations (full, left half and right half)
				equalizeHist(aligned_face, aligned_face);
				equalizeHist(leftFace, leftFace);
				equalizeHist(leftFace, leftFace);

				int h = aligned_face.rows;
				int w = aligned_face.cols;
				int midX = w / 2;

				for (int y = 0; y<h; y++) {
					for (int x = 0; x<w; x++) {
						int v;
						if (x < w / 4) {
							v = leftFace.at<uchar>(y, x);
						}
						else if (x < w * 2 / 4) {
							int lv = leftFace.at<uchar>(y, x);
							int wv = aligned_face.at<uchar>(y, x);

							float f = (x - w * 1 / 4) / (float)(w*0.25f);
							v = cvRound((1.0f - f) * lv + (f)* wv);
						}
						else if (x < w * 3 / 4) {
							int rv = rightFace.at<uchar>(y, x - midX);
							int wv = aligned_face.at<uchar>(y, x);

							float f = (x - w * 2 / 4) / (float)(w*0.25f);
							v = cvRound((1.0f - f) * wv + (f)* rv);
						}
						else {
							v = rightFace.at<uchar>(y, x - midX);
						}
						aligned_face.at<uchar>(y, x) = v;
					}
				}

				//filter operation
				Mat filtered = Mat(aligned_face.size(), CV_8U);
				bilateralFilter(aligned_face, filtered, 0, 20.0, 2.0);

				//ellipse size ratios
				const double FACE_ELLIPSE_CY = 0.40;
				const double FACE_ELLIPSE_W = 0.50;
				const double FACE_ELLIPSE_H = 0.80;

				//eliptical mask operation
				Mat mask = Mat(filtered.size(), CV_8U, Scalar(0));
				Point faceCenter = Point(filtered.cols / 2, cvRound(filtered.rows * FACE_ELLIPSE_CY));
				Size size = Size(cvRound(filtered.cols * FACE_ELLIPSE_W), cvRound(filtered.rows * FACE_ELLIPSE_H));
				ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);

				//final face
				Mat final_face = Mat(filtered.size(), CV_8U, Scalar(128));
				filtered.copyTo(final_face, mask);

				//show processed face
#ifdef SHOW_DETECTED_FACE
				imshow(MINI_WINDOW_NAME, final_face);
#endif
				
				//confidence, label, person name and color for drawing text on preview window
				double confidence = 0;
				int label;
				string personName;
				Scalar color = NO_MATCH_COLOR;

				//if the person is recognized
				if (pr.recognize(aligned_face, label, personName, confidence)) {

					color = MATCH_COLOR;
					has_match = true;
					match_conf = confidence;
				}

				//draw circle around faces
				Point center(face_r->x + face_r->width * 0.5, face_r->y + face_r->height * 0.5);
				circle(frame, center, FACE_RADIUS_RATIO * face_r->width, color, CIRCLE_THICKNESS, LINE_TYPE, 0);

				//draw person names
				Point text(face_r->x + face_r->width / 3, face_r->y + 40);
				putText(frame, personName, text, FONT_HERSHEY_PLAIN, 1.5, color, 1.5, CV_AA);
			}
		}

		//draw optional informations
		putText(frame, "Face Recognition Demo", POS_TITLE,
			FONT, SCALE_TITLE, FONT_COLOR, THICKNESS_TITLE, LINE_TYPE);

		putText(frame, format("Faces: %d", numberOfFaces), cvPoint(10, frame.rows - 55),
			FONT, 2, FONT_COLOR, 1, LINE_TYPE);

		putText(frame, format("Frame: %d", c), cvPoint(10, frame.rows - 80),
			FONT, 2, FONT_COLOR, 1, LINE_TYPE);

		putText(frame, format("Match: %s", has_match ? "True" : "False"), cvPoint(10, frame.rows - 30),
			FONT, 2, FONT_COLOR, 1, LINE_TYPE);

		putText(frame, format("Confidence: %f", has_match ? match_conf : 0), cvPoint(10, frame.rows - 5),
			FONT, 2, FONT_COLOR, 1, LINE_TYPE);

		//save preview video to disk
#ifdef WRITE_OUTPUT
		fw.write(frame);
#endif

		//show preview window (main window)
#ifdef SHOW_OUTPUT
		imshow(MAIN_WINDOW_NAME, frame);
#endif
	}

	destroyAllWindows();
	return 0;
}