#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 360;
const int MAX_NUM_OBJECTS = 20;
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;

Mat inputCam;
Mat HSVImage;
Mat thresholdImage;
Mat erodeElement;
Mat dilateElement;

void startCamera(int);
void onTrackbar(int, void*);
void controlWindow (void);
void printValues(void);
void morphOps(Mat&);
void drawObject(int, int, Mat&);
void trackFilteredObject(int&, int&, Mat, Mat&);
string intToString(int);

void startCamera(int cam1) {
  int x=0, y=0;
  VideoCapture cam(cam1);
  if (!cam.isOpened()) {
    cout << "Error: Camera not found or there is some problem connecting it. Please check your camera." << endl;
    exit(-1);
  }
  while(true) {
    cam >> inputCam;
    resize(inputCam, inputCam, Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, INTER_CUBIC);
    cvtColor(inputCam, HSVImage, COLOR_BGR2HSV);
    //inRange(HSVImage,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),thresholdImage);
    inRange(HSVImage,Scalar(60,92,216),Scalar(71,157,256),thresholdImage);
    morphOps(thresholdImage);
    imshow("HSV Image", HSVImage);
    imshow("Threshold Image", thresholdImage);
    trackFilteredObject(x,y,thresholdImage,inputCam);
    imshow("Cam Input", inputCam);
    int c = cvWaitKey(40);
    if(27 == char(c)) break;
    if(32 == char(c)) printValues();
  }
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

  putText(frame,intToString(x)+","+intToString(y),Point(x,y+30),1,1,Scalar(0,255,0),2);
}

void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed){

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

void onTrackbar(int _val, void* _data) {

}

void controlWindow (void) {
  namedWindow("Controls", 0);
  createTrackbar("H Minimum", "Controls", &H_MIN, H_MAX, onTrackbar);
  createTrackbar("H Maximum", "Controls", &H_MAX, H_MAX, onTrackbar);
  createTrackbar("S Minimum", "Controls", &S_MIN, S_MAX, onTrackbar);
  createTrackbar("S Maximum", "Controls", &S_MAX, S_MAX, onTrackbar);
  createTrackbar("V Minimum", "Controls", &V_MIN, V_MAX, onTrackbar);
  createTrackbar("V Maximum", "Controls", &V_MAX, V_MAX, onTrackbar);
}

void printValues(void) {
  cout << "H Min = " << H_MIN << endl;
  cout << "H Max = " << H_MAX << endl;
  cout << "S Min = " << S_MIN << endl;
  cout << "S Max = " << S_MAX << endl;
  cout << "V Min = " << V_MIN << endl;
  cout << "V Max = " << V_MAX << endl << endl;
}

int main(int argc, char* argv[])
{
  //controlWindow();
  startCamera(0);
  destroyAllWindows();
  return 0;
}
