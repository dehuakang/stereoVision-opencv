#include "stereoCam.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdlib.h>

using namespace cv;
using namespace std;

const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 360;
const int MAX_NUM_OBJECTS = 20;
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;

int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

Mat input1, input2;
Mat img1R, img2R;
Mat hsv1, hsv2;
Mat threshold1, threshold2;
Mat erodeElement1, erodeElement2;
Mat dilateElement1, dilateElement2;
Mat M1, D1, M2, D2, R, T, R1, P1, R2, P2, map11, map12, map21, map22;
Rect roi1, roi2;
float zero [16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
Mat Q = Mat(4, 4, CV_32F, zero);

float x1=0, x2=0, z1=0, z2=0;
float X, Y, Z;

void startCamera(int, int);
void displayVideo(void);
int readInCameraParameters(string, string);
void correctImages(void);
void morphOps(Mat&);
void drawObject(int, int, Mat&);
void trackFilteredObject(int, int, Mat, Mat);
string intToString(int);
int calculateLocation(float, float, float, float);

void startCamera(int cam1, int cam2) {
  VideoCapture camLeft(cam1), camRight(cam2);
  if (!camLeft.isOpened() || !camRight.isOpened()) {
    cout << "Error: Stereo Cameras not found or there is some problem connecting them. Please check your cameras.\n";
    exit(-1);
  }
  while(true) {
    camLeft >> input1;
    camRight >> input2;
    resize(input1, input1, Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, INTER_CUBIC);
    resize(input2, input2, Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, INTER_CUBIC);
    if ((input1.rows != input2.rows) || (input1.cols != input2.cols)) {
      cout << "Error: Images from both cameras are not of some size. Please check the size of each camera.\n";
      exit(-1);
    }
    correctImages();
    cvtColor(img1R, hsv1, COLOR_BGR2HSV);
    cvtColor(img2R, hsv2, COLOR_BGR2HSV);
    //inRange(HSVImage,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),thresholdImage);
    inRange(hsv1, Scalar(60,92,216), Scalar(71,157,256), threshold1);
    inRange(hsv2, Scalar(60,92,216), Scalar(71,157,256), threshold2);
    morphOps(threshold1);
    morphOps(threshold2);
    trackFilteredObject(x1, z1, threshold1, img1R);
    trackFilteredObject(x2, z2, threshold2, img2R);
    calculateLocation(x1, z1, x2, z2);
    displayVideo();
    int c = cvWaitKey(40); //wait for 40 milliseconds
    if(27 == char(c)) {
      destroyAllWindows();
      break; //exit the loop if user press "Esc" key  (ASCII value of "Esc" is 27)
    }
    if(32 == char(c)) {
      imwrite("cam1.png", img1R);
      imwrite("cam2.png", img2R);
    }
  } //while loop
}

void displayVideo() {
  imshow("Cam 1 Image", input1);
  imshow("Cam 2 Image", input2);
  imshow("Cam 1 HSV", hsv1);
  imshow("Cam 2 HSV", hsv2);
  imshow("Cam 1 Threshold", threshold1);
  imshow("Cam 2 Threshold", threshold2);
  imshow("Cam 1 Rectified", img1R);
  imshow("Cam 2 Rectified", img2R);
}

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
    return -2;
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

void correctImages() {
  Size img_size = input1.size();
  stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );
  initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
  initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
  remap(input1, img1R, map11, map12, INTER_LINEAR);
  remap(input2, img2R, map21, map22, INTER_LINEAR);
}

void morphOps(Mat &thresh) {
  Mat erodeElement = getStructuringElement( MORPH_RECT,Size(3,3));
  Mat dilateElement = getStructuringElement( MORPH_RECT,Size(8,8));
  erode(thresh,thresh,erodeElement);
  erode(thresh,thresh,erodeElement);
  dilate(thresh,thresh,dilateElement);
  dilate(thresh,thresh,dilateElement);
}

void drawObject(int x, int y, Mat &frame) {
  circle(frame,Point(x,y),20,Scalar(0,255,0),2);
    if(y-25>0)
    line(frame,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,0),Scalar(0,255,0),2);
    if(y+25<FRAME_HEIGHT)
    line(frame,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,FRAME_HEIGHT),Scalar(0,255,0),2);
    if(x-25>0)
    line(frame,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(0,y),Scalar(0,255,0),2);
    if(x+25<FRAME_WIDTH)
    line(frame,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(FRAME_WIDTH,y),Scalar(0,255,0),2);

  putText(frame,intToString(Z)+"  ,"+intToString(x)+","+intToString(y),Point(x,y+30),1,1,Scalar(0,255,0),2);
}

void trackFilteredObject(int x, int y, Mat threshold, Mat cameraFeed){
  Mat temp;
  threshold.copyTo(temp);
  vector< vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
  double refArea = 0;
  bool objectFound = false;
  if (hierarchy.size() > 0) {
    int numObjects = hierarchy.size();
    if(numObjects < 20) {
      for (int index = 0; index >= 0; index = hierarchy[index][0]) {
        Moments moment = moments((cv::Mat)contours[index]);
        double area = moment.m00;
        if(area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
          x = moment.m10/area;
          y = moment.m01/area;
          objectFound = true;
          refArea = area;
        }
        else {
          objectFound = false;
        }
      }
      if(objectFound ==true) {
        putText(cameraFeed,"Tracking Object",Point(0,50),2,1,Scalar(0,255,0),2);
        drawObject(x,y,cameraFeed);}
    }
    else {
      putText(cameraFeed,"TOO MUCH NOISE! ADJUST FILTER",Point(0,50),1,2,Scalar(0,0,255),2);
    }
  }
}

string intToString(int number) {
  stringstream ss;
  ss << number;
  return ss.str();
}

int calculateLocation(float x1i, float y1i, float x2i, float y2i) {
  int d = x1i - x2i;
  float xi = x2i * Q.at<float>(0, 0) + Q.at<float>(0, 3);
  float yi = y2i * Q.at<float>(1, 1) + Q.at<float>(1, 3);
  float zi = Q.at<float>(2, 3);
  float wi = d * Q.at<float>(3, 2) + Q.at<float>(3, 3);

  X = xi / wi;
  Y = yi / wi;
  Z = zi / wi;
}

int main(int argc, char** argv)
{
  const String keys =
    "{cam1|1|Camera 1 Index}"
    "{cam2|2|Camera 2 Index}"
    "{i|0|Extrinsic File}"
    "{e|0|Intrinsic File}";
  CommandLineParser parser(argc, argv, keys);

  readInCameraParameters(parser.get<string>("i"), parser.get<string>("e"));

  //Start cameras and display video
  startCamera(parser.get<int>("cam1"), parser.get<int>("cam2"));
  return 0;
}
