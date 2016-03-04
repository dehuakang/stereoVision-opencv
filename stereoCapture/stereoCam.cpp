#include "stereoCam.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdlib.h>
#include <math.h>

using namespace cv;
using namespace std;

//const int FRAME_WIDTH = 640;
//const int FRAME_HEIGHT = 360;
const int FRAME_WIDTH = 1280;
const int FRAME_HEIGHT = 720;
const int MAX_NUM_OBJECTS = 20;
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;

const int MAX = 255;
int H_MIN1 = 60;
int H_MAX1 = 71;
int S_MIN1 = 92;
int S_MAX1 = 157;
int V_MIN1 = 216;
int V_MAX1 = 255;
int H_MIN2 = 60;
int H_MAX2 = 71;
int S_MIN2 = 92;
int S_MAX2 = 157;
int V_MIN2 = 216;
int V_MAX2 = 255;
int ERODE_VAL = 1;
int DILATE_VAL = 6;

Mat input1, input2;
Mat hsv1, hsv2;
Mat threshold1, threshold2;

Mat M1, D1, M2, D2, R, T, R1, P1, R2, P2, Q;
Rect roi1, roi2;

int xLoc1=0, xLoc2=0, yLoc1=0, yLoc2=0;
double D = 0;

int readInCameraParameters(string, string);
void displayVideo(string, Mat);
void correctStereoImages(Mat&, Mat&);
void morphOps(Mat&);
void drawTarget(int, int, Mat&);
void trackFilteredObject(int&, int&, Mat, Mat);
string intToString(int);
string doubleToString(double);
void calculateLocation(void);
void onTrackbar(int, void*);
void controlWindow (void);

int readInCameraParameters(string intrinsic_filename, string extrinsic_filename) {
  FileStorage fs(intrinsic_filename, FileStorage::READ);
  if(!fs.isOpened()) {
    cout << "Failed to open file " << intrinsic_filename << endl;
    return -1;
  }

  fs["M1"] >> M1;
  fs["D1"] >> D1;
  fs["M2"] >> M2;
  fs["D2"] >> D2;

  fs.open(extrinsic_filename, FileStorage::READ);
  if(!fs.isOpened()) {
    cout << "Failed to open file " << extrinsic_filename << endl;
    return -1;
  }

  fs["R"] >> R;
  fs["T"] >> T;
  fs["R1"] >> R1;
  fs["R2"] >> R2;
  fs["P1"] >> P1;
  fs["P2"] >> P2;
  fs["Q"] >> Q;

  return 0;
}

void displayVideo(string name, Mat image) {
  imshow(name, image);
}

void correctStereoImages(Mat &cameraFeed1, Mat &cameraFeed2) {
  Mat map11, map12, map21, map22;
  Size img_size = cameraFeed1.size();
  stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );
  initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
  initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
  remap(cameraFeed1, cameraFeed1, map11, map12, INTER_LINEAR);
  remap(cameraFeed2, cameraFeed2, map21, map22, INTER_LINEAR);
}

void morphOps(Mat &thresh) {
  Mat erodeElement = getStructuringElement(MORPH_RECT, Size(ERODE_VAL, ERODE_VAL));
  Mat dilateElement = getStructuringElement(MORPH_RECT, Size(DILATE_VAL, DILATE_VAL));
  erode(thresh, thresh, erodeElement);
  erode(thresh, thresh, erodeElement);
  dilate(thresh, thresh, dilateElement);
  dilate(thresh, thresh, dilateElement);
}

void drawTarget(int x, int y, Mat &cameraFeed) {
  circle(cameraFeed, Point(x, y), 20, Scalar(0, 255, 0), 2);
  if(y-25 > 0) {
    line(cameraFeed, Point(x, y), Point(x, y-25), Scalar(0, 255, 0), 2);
  } else {
      line(cameraFeed, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
  }

  if(y+25 < FRAME_HEIGHT) {
    line(cameraFeed, Point(x, y), Point(x, y+25), Scalar(0, 255, 0), 2);
  } else {
      line(cameraFeed, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
  }

  if(x-25 > 0) {
    line(cameraFeed, Point(x, y), Point(x-25, y), Scalar(0, 255, 0), 2);
  } else {
      line(cameraFeed, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
  }

  if(x+25 < FRAME_WIDTH) {
    line(cameraFeed, Point(x, y), Point(x+25, y), Scalar(0, 255, 0), 2);
  } else {
      line(cameraFeed, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);
  }

  putText(cameraFeed, intToString(x)+","+intToString(y), Point(x, y+30), 1, 1, Scalar(0, 255, 0), 2);
}

void trackFilteredObject(int &x, int &y, Mat threshold, Mat cameraFeed) {
  Mat temp;
  threshold.copyTo(temp);
  vector< vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
  double refArea = 0;
  bool objectFound = false;
  if (hierarchy.size() > 0) {
    int numObjects = hierarchy.size();
    if(numObjects < MAX_NUM_OBJECTS) {
      for (int index = 0; index >= 0; index = hierarchy[index][0]) {
        Moments moment = moments((cv::Mat)contours[index]);
        double area = moment.m00;
        if(area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
          x = moment.m10/area;
          y = moment.m01/area;
          objectFound = true;
          refArea = area;
        } else {
            objectFound = false;
        }
      }
      if(objectFound ==true) {
        putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
        putText(cameraFeed, doubleToString(D), Point(300, 310), 2, 1, Scalar(0, 0, 255), 2);
        drawTarget(x, y, cameraFeed);
      } else {
          putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0 ,0 ,255), 2);
      }
    }
  }
}

string intToString(int number) {
  stringstream ss;
  ss << number;
  return ss.str();
}

string doubleToString(double number) {
  stringstream ss;
  ss << number;
  return ss.str();
}

void calculateLocation(void) {
  D = (double)(50 * FRAME_WIDTH) / (double)(2 * tan(72/2) * (abs(xLoc1 - xLoc2)-26));
}

void onTrackbar(int _val, void* _data) {

}

void controlWindow (void) {
  namedWindow("Controls", 0);
  createTrackbar("Image 1 H Min", "Controls", &H_MIN1, MAX, onTrackbar);
  createTrackbar("Image 1 H Max", "Controls", &H_MAX1, MAX, onTrackbar);
  createTrackbar("Image 1 S Min", "Controls", &S_MIN1, MAX, onTrackbar);
  createTrackbar("Image 1 S Max", "Controls", &S_MAX1, MAX, onTrackbar);
  createTrackbar("Image 1 V Min", "Controls", &V_MIN1, MAX, onTrackbar);
  createTrackbar("Image 1 V Max", "Controls", &V_MAX1, MAX, onTrackbar);
  createTrackbar("Image 2 H Min", "Controls", &H_MIN2, MAX, onTrackbar);
  createTrackbar("Image 2 H Max", "Controls", &H_MAX2, MAX, onTrackbar);
  createTrackbar("Image 2 S Min", "Controls", &S_MIN2, MAX, onTrackbar);
  createTrackbar("Image 2 S Max", "Controls", &S_MAX2, MAX, onTrackbar);
  createTrackbar("Image 2 V Min", "Controls", &V_MIN2, MAX, onTrackbar);
  createTrackbar("Image 2 V Max", "Controls", &V_MAX2, MAX, onTrackbar);
  createTrackbar("Erode Value", "Controls", &ERODE_VAL, 15, onTrackbar);
  createTrackbar("Dilate Value", "Controls", &DILATE_VAL, 25, onTrackbar);
}

int main(int argc, char** argv)
{
  bool showCalibrationWindow = false;
  int c = 0;

  const String keys =
    "{cam1|1|Camera 1 Index}"
    "{cam2|2|Camera 2 Index}"
    "{i|0|Extrinsic File}"
    "{e|0|Intrinsic File}";
  CommandLineParser parser(argc, argv, keys);

  //Read in intrinsic and extrinsic files for cameras
  if(readInCameraParameters(parser.get<string>("i"), parser.get<string>("e"))) {
    cout << "Error: The files containing the parameters for the camera calibration were not found or there were errors in them." << endl;
    exit(-1);
  }

  //Start the camera captures
  VideoCapture camLeft(parser.get<int>("cam1")), camRight(parser.get<int>("cam2"));
  if (!camLeft.isOpened() || !camRight.isOpened()) {
    cout << "Error: Stereo Cameras not found or there is some problem connecting them. Please check your cameras." << endl;
    exit(-1);
  }

  //Main loop
  while(true) {
    camLeft >> input1;
    camRight >> input2;
    if ((input1.rows != input2.rows) || (input1.cols != input2.cols)) {
      cout << "Error: Images from both cameras are not of some size. Please check the size of each camera." << endl;
      exit(-1);
    }
    //resize(input1, input1, Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, INTER_CUBIC);
    //resize(input2, input2, Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, INTER_CUBIC);
    correctStereoImages(input1, input2);
    cvtColor(input1, hsv1, COLOR_BGR2HSV);
    cvtColor(input2, hsv2, COLOR_BGR2HSV);
    inRange(hsv1, Scalar(H_MIN1, S_MIN1, V_MIN1), Scalar(H_MAX1, S_MAX1, V_MAX1), threshold1);
    inRange(hsv2, Scalar(H_MIN2, S_MIN2, V_MIN2), Scalar(H_MAX2, S_MAX2, V_MAX2), threshold2);
    morphOps(threshold1);
    morphOps(threshold2);
    trackFilteredObject(xLoc1, yLoc1, threshold1, input1);
    trackFilteredObject(xLoc2, yLoc2, threshold2, input2);
    calculateLocation();
    displayVideo("Input1 Video", input1);
    displayVideo("Input2 Video", input2);
    if(showCalibrationWindow) {
      controlWindow();
      displayVideo("HSV 1 Video", hsv1);
      displayVideo("HSV 2 Video", hsv2);
      displayVideo("Threshold 1 Video", threshold1);
      displayVideo("Threshold 2 Video", threshold2);
      c = 0;
    } else {
      destroyWindow("Controls");
      destroyWindow("HSV 1 Video");
      destroyWindow("HSV 2 Video");
      destroyWindow("Threshold 1 Video");
      destroyWindow("Threshold 2 Video");
    }
    c = cvWaitKey(40); //wait for 40 milliseconds
    if(27 == char(c)) {
      destroyAllWindows();
      break; //exit the loop if user press "Esc" key  (ASCII value of "Esc" is 27)
    }
    if(32 == char(c)) {
      imwrite("cam1.png", input1);
      imwrite("cam2.png", input2);
    }
    if(67 == char(c) || 99 == char(c)) {
      showCalibrationWindow = !showCalibrationWindow;
    }
  } //Main while loop

  return 0;
}
