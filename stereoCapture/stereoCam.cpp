#include "stereoCam.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat inputLeft, inputRight;
Mat cam1map1, cam1map2;
Mat cam2map1, cam2map2;
Mat leftStereoUndistorted, rightStereoUndistorted;

void startCamera(int, int);
void displayVideo(void);

void startCamera(int cam1, int cam2) {
  VideoCapture camLeft(cam1), camRight(cam2);
  if (!camLeft.isOpened() || !camRight.isOpened()) {
    cout << "Error: Stereo Cameras not found or there is some problem connecting them. Please check your cameras.\n";
    exit(-1);
  }
  while(true) {
    camLeft >> inputLeft;
    camRight >> inputRight;
    resize(inputLeft, inputLeft, Size(640, 360), 0, 0, INTER_CUBIC);
    resize(inputRight, inputRight, Size(640, 360), 0, 0, INTER_CUBIC);
    if ((inputLeft.rows != inputRight.rows) || (inputLeft.cols != inputRight.cols)) {
      cout << "Error: Images from both cameras are not of some size. Please check the size of each camera.\n";
      exit(-1);
    }
    displayVideo();
    int c = cvWaitKey(40); //wait for 40 milliseconds
    if(27 == char(c)) break; //exit the loop if user press "Esc" key  (ASCII value of "Esc" is 27)
    if(32 == char(c)) {
      imwrite("test_right.png", inputRight);
      imwrite("test_left.png", inputLeft);
    }
  } //while loop
}

void displayVideo() {
  imshow("Left Image", inputLeft);
  namedWindow("Left Image", 0);
  imshow("Right Image", inputRight);
  namedWindow("Right Image", 0);
}

int main(int argc, char** argv)
{
  const String keys =
    "{cam1|1|Camera 1 Index}"
    "{cam2|2|Camera 2 Index}";
  CommandLineParser parser(argc, argv, keys);

  //Start cameras and display video
  startCamera(parser.get<int>("cam1"), parser.get<int>("cam2"));
  return 0;
}
