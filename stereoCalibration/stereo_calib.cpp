bplist00—_WebMainResource’	
_WebResourceData_WebResourceMIMEType_WebResourceTextEncodingName_WebResourceFrameName^WebResourceURLOKc<html><head></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">/* *************** Stereo Camera Calibration **************************
 This code can be used to calibrate stereo cameras to get the intrinsic
 and extrinsic files.
 This code also generated rectified image, and also shows RMS Error and Reprojection error
 to find the accuracy of calibration.
 You can load saved stereo images or use this code to capture them in real time.
 Keyboard Shortcuts for real time (ie clicking stereo image at run time):
 1. Default Mode: Detecting (Which detects chessboard corners in real time)
 2. 'c': Starts capturing stereo images (With 2 Sec gap, This can be changed by changing 'timeGap' macro)
 3. 'p': Process and Calibrate (Once all the images are clicked you can press 'p' to calibrate)
 Usage: StereoCameraCallibration [params]
 --cam1 (value:0)                           Camera 1 Index
 --cam2 (value:2)                           Camera 2 Index
 --dr, --folder (value:.)                   Directory of images
 -h, --height (value:6)                     Height of the board
 --help (value:true)                        Prints this
 --images, -n (value:40)                    No of stereo pair images
 --post, --postfix (value:jpg)              Image extension. Ex: jpg,png etc
 --prefixleft, --prel (value:image_left_)   Left image name prefix. Ex: image_left_
 --prefixright, --prer (value:image_right_) Right image name postfix. Ex: image_right_
 --realtime, --rt (value:1)                 Clicks stereo images before calibration. Use if you do not have stereo pair images saved
 -w, --width (value:7)                      Width of the board
 Example:   ./stereo_calib                                              Clicks stereo images at run time.
 ./stereo_calib -rt=0 -prel=left_ -prer=right_ -post=jpg     RealTime id off ie images should be loaded from disk. With images named left_1.jpg, right_1.jpg etc.
 Cheers
 Abhishek Upperwal
 ***********************************************************************/
/* *************** License:**************************
 By downloading, copying, installing or using the software you agree to this license.
 If you do not agree to this license, do not download, install, copy or use the software.
 License Agreement
 For Open Source Computer Vision Library
 (3-clause BSD License)
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 Neither the names of the copyright holders nor the names of the contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 This software is provided by the copyright holders and contributors ‚Äúas is‚Äù and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall copyright holders or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
 ************************************************** */

/* ************* Original reference:**************
 Oct. 3, 2008
 BOOK:It would be nice if you cited it:
 Learning OpenCV: Computer Vision with the OpenCV Library
 by Gary Bradski and Adrian Kaehler
 Published by O'Reilly Media, October 3, 2008
 AVAILABLE AT:
 http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
 Or: http://oreilly.com/catalog/9780596516130/
 ISBN-10: 0596516134 or: ISBN-13: 978-0596516130
 OPENCV WEBSITES:
 Homepage:      http://opencv.org
 Online docs:   http://docs.opencv.org
 Q&amp;A forum:     http://answers.opencv.org
 Issue tracker: http://code.opencv.org
 GitHub:        https://github.com/Itseez/opencv/
 ************************************************** */

#include &lt;opencv2/calib3d/calib3d.hpp&gt;
#include &lt;opencv2/imgproc/imgproc.hpp&gt;
#include &lt;opencv2/highgui/highgui.hpp&gt;
#include &lt;opencv2/imgcodecs.hpp&gt;
#include &lt;iostream&gt;
#include &lt;stdlib.h&gt;

#define timeGap 3000000000U

using namespace cv;
using namespace std;

static void help() {
    cout&lt;&lt;"/******** HELP *******/\n";
    cout &lt;&lt; "\nThis program helps you to calibrate the stereo cameras.\n This program generates intrinsics.yml and extrinsics.yml which can be used in Stereo Matching Algorithms.\n";
    cout&lt;&lt;"It also displays the rectified image\n";
    cout&lt;&lt;"\nKeyboard Shortcuts for real time (ie clicking stereo image at run time):\n";
    cout&lt;&lt;"1. Default Mode: Detecting (Which detects chessboard corners in real time)\n";
    cout&lt;&lt;"2. 'c': Starts capturing stereo images (With 2 Sec gap, This can be changed by changing 'timeGap' macro)\n";
    cout&lt;&lt;"3. 'p': Process and Calibrate (Once all the images are clicked you can press 'p' to calibrate)";
    cout&lt;&lt;"\nType ./stereo_calib --help for more details.\n";
    cout&lt;&lt;"\n/******* HELP ENDS *********/\n\n";
}

enum Modes { DETECTING, CAPTURING, CALIBRATING};
Modes mode = DETECTING;
int noOfStereoPairs;
int stereoPairIndex = 0, cornerImageIndex=0;
int goIn = 1;
Mat _leftOri, _rightOri;
int64 prevTickCount;
vector&lt;Point2f&gt; cornersLeft, cornersRight;
vector&lt;vector&lt;Point2f&gt; &gt; cameraImagePoints[2];
Size boardSize;

string prefixLeft;
string prefixRight;
string postfix;
string dir;

int calibType;

Mat displayCapturedImageIndex(Mat);
Mat displayMode(Mat);
bool findChessboardCornersAndDraw(Mat, Mat);
void displayImages();
void saveImages(Mat, Mat, int);
void calibrateStereoCamera(Size);
void calibrateInRealTime(int, int);
void calibrateFromSavedImages(string, string, string, string);

Mat displayCapturedImageIndex(Mat img) {
    std::ostringstream imageIndex;
    imageIndex&lt;&lt;stereoPairIndex&lt;&lt;"/"&lt;&lt;noOfStereoPairs;
    putText(img, imageIndex.str().c_str(), Point(50, 70), FONT_HERSHEY_PLAIN, 0.9, Scalar(0,0,255), 2);
    return img;
}

Mat displayMode(Mat img) {
    String modeString = "DETECTING";
    if (mode == CAPTURING) {
        modeString="CAPTURING";
    }
    else if (mode == CALIBRATING) {
        modeString="CALIBRATED";
    }
    putText(img, modeString, Point(50,50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
    if (mode == CAPTURING) {
        img = displayCapturedImageIndex(img);
    }
    return img;
}

bool findChessboardCornersAndDraw(Mat inputLeft, Mat inputRight) {
    _leftOri = inputLeft;
    _rightOri = inputRight;
    bool foundLeft = false, foundRight = false;
    cvtColor(inputLeft, inputLeft, COLOR_BGR2GRAY);
    cvtColor(inputRight, inputRight, COLOR_BGR2GRAY);
    foundLeft = findChessboardCorners(inputLeft, boardSize, cornersLeft, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    foundRight = findChessboardCorners(inputRight, boardSize, cornersRight, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    drawChessboardCorners(_leftOri, boardSize, cornersLeft, foundLeft);
    drawChessboardCorners(_rightOri, boardSize, cornersRight, foundRight);
    _leftOri = displayMode(_leftOri);
    _rightOri = displayMode(_rightOri);
    if (foundLeft &amp;&amp; foundRight) {
        return true;
    }
    else {
        return false;
    }
}

void displayImages() {
    imshow("Left Image", _leftOri);
    imshow("Right Image", _rightOri);
}

void saveImages(Mat leftImage, Mat rightImage, int pairIndex) {
    cameraImagePoints[0].push_back(cornersLeft);
    cameraImagePoints[1].push_back(cornersRight);
    if (calibType == 1) {
        cvtColor(leftImage, leftImage, COLOR_BGR2GRAY);
        cvtColor(rightImage, rightImage, COLOR_BGR2GRAY);
        std::ostringstream leftString, rightString;
        leftString&lt;&lt;dir&lt;&lt;"/"&lt;&lt;prefixLeft&lt;&lt;pairIndex&lt;&lt;postfix;
        rightString&lt;&lt;dir&lt;&lt;"/"&lt;&lt;prefixRight&lt;&lt;pairIndex&lt;&lt;postfix;
        imwrite(leftString.str().c_str(), leftImage);
        imwrite(rightString.str().c_str(), rightImage);
    }
}

void calibrateStereoCamera(Size imageSize) {
    vector&lt;vector&lt;Point3f&gt; &gt; objectPoints;
    objectPoints.resize(noOfStereoPairs);
    for (int i=0; i&lt;noOfStereoPairs; i++) {
        for (int j=0; j&lt;boardSize.height; j++) {
            for (int k=0; k&lt;boardSize.width; k++) {
                objectPoints[i].push_back(Point3f(float(k),float(j),0.0));
            }
        }
    }
    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;
    double rms = stereoCalibrate(objectPoints, cameraImagePoints[0], cameraImagePoints[1],
                                 cameraMatrix[0], distCoeffs[0],
                                 cameraMatrix[1], distCoeffs[1],
                                 imageSize, R, T, E, F,
                                 CALIB_FIX_ASPECT_RATIO +
                                 CALIB_ZERO_TANGENT_DIST +
                                 CALIB_SAME_FOCAL_LENGTH +
                                 CALIB_RATIONAL_MODEL +
                                 CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                                 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    cout&lt;&lt;"RMS Error: "&lt;&lt;rms&lt;&lt;"\n";
    double err = 0;
    int npoints = 0;
    vector&lt;Vec3f&gt; lines[2];
    for(int i = 0; i &lt; noOfStereoPairs; i++ )
    {
        int npt = (int)cameraImagePoints[0][i].size();
        Mat imgpt[2];
        for(int k = 0; k &lt; 2; k++ )
        {
            imgpt[k] = Mat(cameraImagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for(int j = 0; j &lt; npt; j++ )
        {
            double errij = fabs(cameraImagePoints[0][i][j].x*lines[1][j][0] +
                                cameraImagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
            fabs(cameraImagePoints[1][i][j].x*lines[0][j][0] +
                 cameraImagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout &lt;&lt; "Average Reprojection Error: " &lt;&lt;  err/npoints &lt;&lt; endl;
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs &lt;&lt; "M1" &lt;&lt; cameraMatrix[0] &lt;&lt; "D1" &lt;&lt; distCoeffs[0] &lt;&lt;
        "M2" &lt;&lt; cameraMatrix[1] &lt;&lt; "D2" &lt;&lt; distCoeffs[1];
        fs.release();
    }
    else
        cout&lt;&lt;"Error: Could not open intrinsics file.";
    Mat R1, R2, P1, P2, Q;
    Rect validROI[2];
    stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 1, imageSize, &amp;validROI[0], &amp;validROI[1]);
    fs.open("extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs &lt;&lt; "R" &lt;&lt; R &lt;&lt; "T" &lt;&lt; T &lt;&lt; "R1" &lt;&lt; R1 &lt;&lt; "R2" &lt;&lt; R2 &lt;&lt; "P1" &lt;&lt; P1 &lt;&lt; "P2" &lt;&lt; P2 &lt;&lt; "Q" &lt;&lt; Q;
        fs.release();
    }
    else
        cout&lt;&lt;"Error: Could not open extrinsics file";
    bool isVerticalStereo = fabs(P2.at&lt;double&gt;(1, 3)) &gt; fabs(P2.at&lt;double&gt;(0, 3));
    Mat rmap[2][2];
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
    Mat canvas;
    double sf;
    int w, h;
    if (!isVerticalStereo) {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }
    String file;
    namedWindow("rectified");
    for (int i=0; i &lt; noOfStereoPairs; i++) {
        for (int j=0; j &lt; 2; j++) {
            if (j==0) {
                file = prefixLeft;
            }
            else if (j==1) {
                file = prefixRight;
            }
            ostringstream st;
            st&lt;&lt;dir&lt;&lt;"/"&lt;&lt;file&lt;&lt;i+1&lt;&lt;"."&lt;&lt;postfix;
            Mat img = imread(st.str().c_str()), rimg, cimg;
            remap(img, rimg, rmap[j][0], rmap[j][1], INTER_LINEAR);
            cimg=rimg;
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*j, 0, w, h)) : canvas(Rect(0, h*j, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            Rect vroi(cvRound(validROI[j].x*sf), cvRound(validROI[j].y*sf),
                      cvRound(validROI[j].width*sf), cvRound(validROI[j].height*sf));
            rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
        }
        if( !isVerticalStereo )
            for(int j = 0; j &lt; canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for(int j = 0; j &lt; canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
    }
}

void calibrateInRealTime(int cam1, int cam2) {
    VideoCapture camLeft(cam1), camRight(cam2);
    if (!camLeft.isOpened() || !camRight.isOpened()) {
        cout&lt;&lt;"Error: Stereo Cameras not found or there is some problem connecting them. Please check your cameras.\n";
        exit(-1);
    }
    Mat inputLeft, inputRight, copyImageLeft, copyImageRight;
    bool foundCornersInBothImage = false;
    for( ; ; ) {
        camLeft&gt;&gt;inputLeft;
        camRight&gt;&gt;inputRight;
        if ((inputLeft.rows != inputRight.rows) || (inputLeft.cols != inputRight.cols)) {
            cout&lt;&lt;"Error: Images from both cameras are not of some size. Please check the size of each camera.\n";
            exit(-1);
        }
        inputLeft.copyTo(copyImageLeft);
        inputRight.copyTo(copyImageRight);
        foundCornersInBothImage = findChessboardCornersAndDraw(inputLeft, inputRight);
        if (foundCornersInBothImage &amp;&amp; mode == CAPTURING &amp;&amp; stereoPairIndex&lt;noOfStereoPairs) {
            int64 thisTick = getTickCount();
            int64 diff = thisTick - prevTickCount;
            if (goIn==1 || diff &gt;= timeGap) {
                goIn=0;
                saveImages(copyImageLeft, copyImageRight, ++stereoPairIndex);
                prevTickCount = getTickCount();
            }
        }
        displayImages();
        if (mode == CALIBRATING) {
            calibrateStereoCamera(inputLeft.size());
            waitKey();
        }
        char keyBoardInput = (char)waitKey(50);
        if (keyBoardInput == 'q' || keyBoardInput == 'Q') {
            exit(-1);
        }
        else if(keyBoardInput == 'c' || keyBoardInput == 'C') {
            mode = CAPTURING;
        }
        else if (keyBoardInput == 'p' || keyBoardInput == 'P') {
            mode = CALIBRATING;
        }
    }
}

void calibrateFromSavedImages(string dr, string prel, string prer, string post) {
    Size imageSize;
    for (int i=0; i&lt;noOfStereoPairs; i++) {
        Mat inputLeft, inputRight, copyImageLeft, copyImageRight;
        ostringstream imgIndex;
        imgIndex &lt;&lt; i+1;
        bool foundCornersInBothImage = false;
        string sourceLeftImagePath, sourceRightImagePath;
        sourceLeftImagePath = dr+"/"+prel+imgIndex.str()+"."+post;
        sourceRightImagePath = dr+"/"+prer+imgIndex.str()+"."+post;
        inputLeft = imread(sourceLeftImagePath);
        inputRight = imread(sourceRightImagePath);
        imageSize = inputLeft.size();
        if (inputLeft.empty() || inputRight.empty()) {
            cout&lt;&lt;"\nCould no find image: "&lt;&lt;sourceLeftImagePath&lt;&lt;" or "&lt;&lt;sourceRightImagePath&lt;&lt;". Skipping images.\n";
            continue;
        }
        if ((inputLeft.rows != inputRight.rows) || (inputLeft.cols != inputRight.cols)) {
            cout&lt;&lt;"\nError: Left and Right images are not of some size. Please check the size of the images. Skipping Images.\n";
            continue;
        }
        inputLeft.copyTo(copyImageLeft);
        inputRight.copyTo(copyImageRight);
        foundCornersInBothImage = findChessboardCornersAndDraw(inputLeft, inputRight);
        if (foundCornersInBothImage &amp;&amp; stereoPairIndex&lt;noOfStereoPairs) {
            saveImages(copyImageLeft, copyImageRight, ++stereoPairIndex);
        }
        displayImages();
    }
    if(stereoPairIndex &gt; 2) {
        calibrateStereoCamera(imageSize);
        waitKey();
    }
    else {
        cout&lt;&lt;"\nInsufficient stereo images to calibrate.\n";
    }
}

int main(int argc, char** argv) {
    help();
    const String keys =
    "{help| |Prints this}"
    "{h height|6|Height of the board}"
    "{w width|7|Width of the board}"
    "{rt realtime|1|Clicks stereo images before calibration. Use if you do not have stereo pair images saved}"
    "{n images|40|No of stereo pair images}"
    "{dr folder|.|Directory of images}"
    "{prel prefixleft|image_left_|Left image name prefix. Ex: image_left_}"
    "{prer prefixright|image_right_|Right image name postfix. Ex: image_right_}"
    "{post postfix|jpg|Image extension. Ex: jpg,png etc}"
    "{cam1|0|Camera 1 Index}"
    "{cam2|2|Camera 2 Index}";
    CommandLineParser parser(argc, argv, keys);
    if(parser.has("help")) {
        parser.printMessage();
        exit(-1);
    }
    boardSize = Size(parser.get&lt;int&gt;("w"), parser.get&lt;int&gt;("h"));
    noOfStereoPairs = parser.get&lt;int&gt;("n");
    prefixLeft = parser.get&lt;string&gt;("prel");
    prefixRight = parser.get&lt;string&gt;("prer");
    postfix = parser.get&lt;string&gt;("post");
    dir =parser.get&lt;string&gt;("dr");
    calibType = parser.get&lt;int&gt;("rt");
    namedWindow("Left Image");
    namedWindow("Right Image");
    switch (calibType) {
        case 0:
            calibrateFromSavedImages(dir, prefixLeft, prefixRight, postfix);
            break;
        case 1:
            calibrateInRealTime(parser.get&lt;int&gt;("cam1"), parser.get&lt;int&gt;("cam2"));
            break;
        default:
            cout&lt;&lt;"-rt should be 0 or 1. Ex: -rt=1\n";
            break;
    }
    return 0;
}</pre></body></html>Ztext/plainUUTF-8P_Uhttps://raw.githubusercontent.com/upperwal/opencv/master/samples/cpp/stereo_calib.cpp    ( : P n Ö îK˚LLL                           Le