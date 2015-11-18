
#include "stdafx.h"

/*using namespace cv;
int main()
{
	IplImage*src=cvLoadImage("3.jpg");
	cvNamedWindow("example");
	cvShowImage("example",src);
	cvWaitKey(6000);
	cvReleaseImage(&src);
}*/
/*#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream> 
using namespace cv;
using namespace std; 
static Mat norm_0_255(InputArray _src) {    Mat src = _src.getMat();    // �����ͷ���һ����һ�����ͼ�����:  
Mat dst;    
switch(src.channels()) {  
case1:      
	cv::normalize(_src, dst, 0,255, NORM_MINMAX, CV_8UC1);      
	break;   
case3:       
	cv::normalize(_src, dst, 0,255, NORM_MINMAX, CV_8UC3);        
	break;   
default:        
	src.copyTo(dst);        break;    }    return dst;}//ʹ��CSV�ļ�ȥ��ͼ��ͱ�ǩ����Ҫʹ��stringstream��getline����
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator =';') {    
	std::ifstream file(filename.c_str(), ifstream::in); 
	if (!file) {        
		string error_message ="No valid input file was given, please check the given filename.";       
		CV_Error(CV_StsBadArg, error_message);  
	}  
	string line, path, classlabel;   
	while (getline(file, line)) {    
		stringstream liness(line);      
		getline(liness, path, separator);     
		getline(liness, classlabel);      
		if(!path.empty()&&!classlabel.empty()) {   
			images.push_back(imread(path, 0));    
			labels.push_back(atoi(classlabel.c_str()));     
		}  
	}
} 
int main(int argc, const char*argv[]) {    
	// ���Ϸ��������ʾ�÷�    // ���û�в����������˳���.  
	if (argc <2) {      
		cout <<"usage: "<< argv[0]<<" <csv.ext> <output_folder> "<< endl;        exit(1);  
	}   
	string output_folder; 
	if (argc ==3) {   
		output_folder = string(argv[2]);  
	}    //��ȡ���CSV�ļ�·��.   
	string fn_csv = string(argv[1]);  
	// 2�����������ͼ�����ݺͶ�Ӧ�ı�ǩ   
	vector<Mat> images; 
	vector<int> labels;  
	// ��ȡ����. ����ļ����Ϸ��ͻ����  
	// ������ļ����Ѿ�����.  
	try {       
		read_csv(fn_csv, images, labels);  
	}
	catch (cv::Exception& e) {      
		cerr <<"Error opening fi      le \""<< fn_csv <<"\". Reason: "<< e.msg << endl;     
		// �ļ������⣬����ɶҲ�������ˣ��˳���    
		exit(1);  
	}   // ���û�ж�ȡ���㹻ͼƬ������Ҳ���˳�. 
	if(images.size()<=1) {    
		string error_message ="This demo needs at least 2 images to work. Please add more images to your data set!";      
		CV_Error(CV_StsError, error_message);    }    // �õ���һ����Ƭ�ĸ߶�. �������ͼ��    // ���ε�����ԭʼ��Сʱ��Ҫ   
	int height = images[0].rows;   
	// ����ļ��д�������Ǵ�������ݼ����Ƴ����һ��ͼƬ    //[gm:��Ȼ������Ҫ�����Լ�����Ҫ�޸ģ���������˺ܶ�����]    
	Mat testSample = images[images.size() -1];    
	int testLabel = labels[labels.size() -1];   
	images.pop_back();    
	labels.pop_back();    // ���漸�д�����һ��������ģ����������ʶ��    // ͨ��CSV�ļ���ȡ��ͼ��ͱ�ǩѵ������    // T������һ��������PCA�任    //�����ֻ�뱣��10�����ɷ֣�ʹ�����´���    //  
	cv::createEigenFaceRecognizer(10);  
	//    // ����㻹ϣ��ʹ�����Ŷ���ֵ����ʼ����ʹ��������䣺    //     
	cv::createEigenFaceRecognizer(10, 123.0);    
	//    // �����ʹ��������������ʹ��һ����ֵ��ʹ��������䣺    //     
	cv::createEigenFaceRecognizer(0, 123.0);    
	//    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();    
	model ->train(images, labels);    // ����Բ���ͼ�����Ԥ�⣬predictedLabel��Ԥ���ǩ���    
	int predictedLabel = model->predict(testSample);    //    // ����һ�ֵ��÷�ʽ�����Ի�ȡ���ͬʱ�õ���ֵ:   
	//    
	int predictedLabel = -1;    //    
	double confidence = 0.0;    //      
	model->predict(testSample, predictedLabel, confidence);    //   
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel); 
	cout << result_message << endl;    // ��������λ�ȡ������ģ�͵�����ֵ�����ӣ�ʹ����getMat����: 
	Mat eigenvalues = model->getMat("eigenvalues");    // ͬ�����Ի�ȡ��������:  
	Mat W = model->getMat("eigenvectors");    // �õ�ѵ��ͼ��ľ�ֵ����   
	Mat mean = model->getMat("mean");    // ��ʵ���Ǳ���:  
	if(argc==2) {  
		imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));   
	} 
	else {     
		imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));  
	}    // ��ʵ���Ǳ���������:  
	for (int i =0; i < min(10, W.cols); i++) {    
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));    
		cout << msg << endl;     
		// �õ��� #i������    
		Mat ev = W.col(i).clone();    
		//�������ԭʼ��С��Ϊ�˰�������ʾ��һ����0~255.       
		Mat grayscale = norm_0_255(ev.reshape(1, height));    
		// ʹ��α��ɫ����ʾ�����Ϊ�˸��õĸ���.     
		Mat cgrayscale;       
		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);   
		// ��ʾ���߱���:      
		if(argc==2) {         
			imshow(format("eigenface_%d", i), cgrayscale);  
		}
		else 
		{            imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));      
		}   
	}    // ��һЩԤ������У���ʾ���Ǳ����ؽ����ͼ��:  
	for(int num_components =10; num_components <300; num_components+=15) {     
		// ��ģ���е�����������ȡһ����     
		Mat evs = Mat(W, Range::all(), Range(0, num_components));      
		Mat projection = subspaceProject(evs, mean, images[0].reshape(1,1));  
		Mat reconstruction = subspaceReconstruct(evs, mean, projection);        // ��һ�������Ϊ����ʾ:    
		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));   
		// ��ʾ���߱���:    
		if(argc==2) {         
			imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);      
		} 
		else 
		{           
			imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);   
		}  
	}    // ������ǲ��Ǵ�ŵ��ļ��У�����ʾ��������ʹ�����ݶ��ȴ���������:    
	if(argc==2) {  
		waitKey(0);
	}   
	return 0;
}        
 */        
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
using namespace cv;
/** Function Headers */
void detectAndDisplay( Mat frame );
/** Global variables */
//-- Note, either copy this file from opencv/data/haarscascades to your current folder, or change these locations
String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Face detection";
/**
* @function main
*/
int main( void )
{
	//-- 1. Load the cascade
	if(!face_cascade.load( face_cascade_name ) )
	{
		printf("--(!)Error loading\n"); return -1; 
	};
	//-- 2. Read the image
	IplImage* img = cvLoadImage("9.jpg", CV_LOAD_IMAGE_COLOR);
	Mat frame(img);
	//-- 3. Apply the classifier to the frame
	if( !frame.empty() )
	{
		detectAndDisplay( frame );
	}
	else
	{
		printf("--(!)Error!\n");
	}
	waitKey();
	cvDestroyWindow(window_name.c_str());
	cvReleaseImage(&img);
	return 0;
}
/**
* @function detectAndDisplay
*/
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	//-- Detect faces
	double t = (double)cvGetTickCount();
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	t = (double)cvGetTickCount() - t; 
	printf("%gms\n", t/((double)cvGetTickFrequency()*1000.0));
	for( size_t i = 0; i < faces.size(); i++ )
	{
		Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 ); 
		printf("Found a face at (%d, %d)\n", center.x, center.y);
		ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 255, 255 ), 2, 8, 0 );
	}
	//-- Show what you got
	imshow( window_name, frame );
}