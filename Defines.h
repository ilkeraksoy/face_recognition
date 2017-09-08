#ifndef DEFINES_H
#define DEFINES_H

//Flags
#define SHOW_OUTPUT
#define SHOW_DETECTED_FACE
#define WRITE_OUTPUT

//Inputs
#define FACE_CASCADE_PATH		"cascades/haarcascade_frontalface_default.xml"
#define LEFT_EYE_CASCADE_PATH   "cascades/haarcascade_mcs_lefteye.xml"
#define RIGHT_EYE_CASCADE_PATH  "cascades/haarcascade_mcs_righteye.xml"
#define INPUT_VIDEO_PATH		"input_video/Obama.mp4"
#define FACES_LIST_PATH			"faces/list.txt"
#define DICTIONARY_PATH			"faces/dictionary.txt"

//Input video
#define START_FRAME  -1
#define END_FRAME    -1
#define FRAMES_DELTA -1

//Output video
#define OUTPUT_VIDEO_PATH   "Output.avi"
#define OUTPUT_VIDEO_FPS    15
#define OUTPUT_VIDEO_FOURCC CV_FOURCC('M', 'J', 'P', 'G') //codec

//Colors, fonts, lines...
#define NO_MATCH_COLOR    Scalar(0, 0, 255) //red
#define MATCH_COLOR       Scalar(0, 255, 0) //green
#define FACE_RADIUS_RATIO 0.5
#define CIRCLE_THICKNESS  2.5
#define LINE_TYPE         CV_AA
#define FONT              FONT_HERSHEY_PLAIN
#define FONT_COLOR        Scalar(0, 255, 0)
#define THICKNESS_TITLE   1.9
#define SCALE_TITLE       1.9
#define POS_TITLE         cvPoint(10, 30)
#define THICKNESS_LINK    1.6
#define SCALE_LINK        1.3
#define POS_LINK          cvPoint(10, 55)

//Face detector
#define FACE_DETECT_SCALE_FACTOR   1.02
#define FACE_DETECT_MIN_NEIGHBORS  40
#define FACE_DETECT_MIN_SIZE	   Size(10, 10)
#define FACE_DETECT_MAX_SIZE	   Size(300, 300)
#define FACE_DETECT_MODE		   0|CV_HAAR_SCALE_IMAGE   //0 //CV_HAAR_DO_CANNY_PRUNING //0    //CASCADE_SCALE_IMAGE //CASCADE_FIND_BIGGEST_OBJECT	

//Face aligner
#define FACE_ALIGN_DESIRED_LEFT_EYE_X  0.16
#define FACE_ALIGN_DESIRED_LEFT_EYE_Y  0.14
#define FACE_ALIGN_FACE_SIZE           Size(150, 150)

//Eye detector
#define EYE_DETECT_SCALE_FACTOR   1.02
#define EYE_DETECT_MIN_NEIGHBORS  40
#define EYE_DETECT_MIN_SIZE		  Size(2, 2)
#define EYE_DETECT_MAX_SIZE		  Size(50, 50)
#define EYE_SX				      0.10
#define EYE_SY                    0.19
#define EYE_SW                    0.40
#define EYE_SH                    0.36

//LBPH face recognizer
#define LBPH_NAME_FILE_PATH  "faces/names.txt"
#define LBPH_YML_FILE_PATH   "faces/persons.yml"
#define LBPH_RADIUS    3
#define LBPH_NEIGHBORS 8
#define LBPH_GRID_X    8
#define LBPH_GRID_Y    8
#define LBPH_THRESHOLD 100.0
#define FACE_SIZE      Size(150, 150)

//Windows
#define MAIN_WINDOW_NAME "FaceRecognitionDemo"
#define MINI_WINDOW_NAME "DetectedFace"

#endif //DEFINES_H