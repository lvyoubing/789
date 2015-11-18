
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
static Mat norm_0_255(InputArray _src) {    Mat src = _src.getMat();    // 创建和返回一个归一化后的图像矩阵:  
Mat dst;    
switch(src.channels()) {  
case1:      
	cv::normalize(_src, dst, 0,255, NORM_MINMAX, CV_8UC1);      
	break;   
case3:       
	cv::normalize(_src, dst, 0,255, NORM_MINMAX, CV_8UC3);        
	break;   
default:        
	src.copyTo(dst);        break;    }    return dst;}//使用CSV文件去读图像和标签，主要使用stringstream和getline方法
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
	// 检测合法的命令，显示用法    // 如果没有参数输入则退出！.  
	if (argc <2) {      
		cout <<"usage: "<< argv[0]<<" <csv.ext> <output_folder> "<< endl;        exit(1);  
	}   
	string output_folder; 
	if (argc ==3) {   
		output_folder = string(argv[2]);  
	}    //读取你的CSV文件路径.   
	string fn_csv = string(argv[1]);  
	// 2个容器来存放图像数据和对应的标签   
	vector<Mat> images; 
	vector<int> labels;  
	// 读取数据. 如果文件不合法就会出错  
	// 输入的文件名已经有了.  
	try {       
		read_csv(fn_csv, images, labels);  
	}
	catch (cv::Exception& e) {      
		cerr <<"Error opening fi      le \""<< fn_csv <<"\". Reason: "<< e.msg << endl;     
		// 文件有问题，我们啥也做不了了，退出了    
		exit(1);  
	}   // 如果没有读取到足够图片，我们也得退出. 
	if(images.size()<=1) {    
		string error_message ="This demo needs at least 2 images to work. Please add more images to your data set!";      
		CV_Error(CV_StsError, error_message);    }    // 得到第一张照片的高度. 在下面对图像    // 变形到他们原始大小时需要   
	int height = images[0].rows;   
	// 下面的几行代码仅仅是从你的数据集中移除最后一张图片    //[gm:自然这里需要根据自己的需要修改，他这里简化了很多问题]    
	Mat testSample = images[images.size() -1];    
	int testLabel = labels[labels.size() -1];   
	images.pop_back();    
	labels.pop_back();    // 下面几行创建了一个特征脸模型用于人脸识别，    // 通过CSV文件读取的图像和标签训练它。    // T这里是一个完整的PCA变换    //如果你只想保留10个主成分，使用如下代码    //  
	cv::createEigenFaceRecognizer(10);  
	//    // 如果你还希望使用置信度阈值来初始化，使用以下语句：    //     
	cv::createEigenFaceRecognizer(10, 123.0);    
	//    // 如果你使用所有特征并且使用一个阈值，使用以下语句：    //     
	cv::createEigenFaceRecognizer(0, 123.0);    
	//    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();    
	model ->train(images, labels);    // 下面对测试图像进行预测，predictedLabel是预测标签结果    
	int predictedLabel = model->predict(testSample);    //    // 还有一种调用方式，可以获取结果同时得到阈值:   
	//    
	int predictedLabel = -1;    //    
	double confidence = 0.0;    //      
	model->predict(testSample, predictedLabel, confidence);    //   
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel); 
	cout << result_message << endl;    // 这里是如何获取特征脸模型的特征值的例子，使用了getMat方法: 
	Mat eigenvalues = model->getMat("eigenvalues");    // 同样可以获取特征向量:  
	Mat W = model->getMat("eigenvectors");    // 得到训练图像的均值向量   
	Mat mean = model->getMat("mean");    // 现实还是保存:  
	if(argc==2) {  
		imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));   
	} 
	else {     
		imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));  
	}    // 现实还是保存特征脸:  
	for (int i =0; i < min(10, W.cols); i++) {    
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));    
		cout << msg << endl;     
		// 得到第 #i个特征    
		Mat ev = W.col(i).clone();    
		//把它变成原始大小，为了把数据显示归一化到0~255.       
		Mat grayscale = norm_0_255(ev.reshape(1, height));    
		// 使用伪彩色来显示结果，为了更好的感受.     
		Mat cgrayscale;       
		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);   
		// 显示或者保存:      
		if(argc==2) {         
			imshow(format("eigenface_%d", i), cgrayscale);  
		}
		else 
		{            imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));      
		}   
	}    // 在一些预测过程中，显示还是保存重建后的图像:  
	for(int num_components =10; num_components <300; num_components+=15) {     
		// 从模型中的特征向量截取一部分     
		Mat evs = Mat(W, Range::all(), Range(0, num_components));      
		Mat projection = subspaceProject(evs, mean, images[0].reshape(1,1));  
		Mat reconstruction = subspaceReconstruct(evs, mean, projection);        // 归一化结果，为了显示:    
		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));   
		// 显示或者保存:    
		if(argc==2) {         
			imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);      
		} 
		else 
		{           
			imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);   
		}  
	}    // 如果我们不是存放到文件中，就显示他，这里使用了暂定等待键盘输入:    
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