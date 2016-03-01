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

Mat input1, input2;
Mat img1R, img2R;
Mat M1, D1, M2, D2, R, T, R1, P1, R2, P2, Q, map11, map12, map21, map22;
Rect roi1, roi2;

void startCamera(int, int);
void displayVideo(void);
int readInCameraParameters(string, string);
void correctImages(void);

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
    displayVideo();
    int c = cvWaitKey(40); //wait for 40 milliseconds
    if(27 == char(c)) {
      destroyAllWindows();
      break; //exit the loop if user press "Esc" key  (ASCII value of "Esc" is 27)
    }
    if(32 == char(c)) {
      imwrite("cam1.png", input1);
      imwrite("cam2.png", input2);
    }
  } //while loop
}

void displayVideo() {
  imshow("Cam 1 Image", input1);
  imshow("Cam 2 Image", input2);
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
