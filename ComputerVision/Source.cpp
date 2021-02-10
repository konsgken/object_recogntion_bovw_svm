#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\ml\ml.hpp>
#include <iostream>
#include "dirent.h"

using namespace cv;
using namespace std;

std::vector<string> getFiles(const char* folder) {
	vector<string> files;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(folder)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			files.push_back(ent->d_name);
			printf("%s\n", ent->d_name);
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
	}
	return files;
}

void train(char* databasePath) {
	string databasePathString = string(databasePath);
	vector<string> files = getFiles(databasePath);
	BOWKMeansTrainer trainer(150);
	SiftFeatureDetector detector = SiftFeatureDetector();
	SiftDescriptorExtractor descriptor = SiftDescriptorExtractor();
	vector<KeyPoint> keypoints;
	for (int i = 2; i < files.size(); i++) {
			string imagePath = databasePathString + "\\" + files[i];
			const char* databasePathString1 = imagePath.c_str();
			vector<string> subfiles = getFiles(databasePathString1);
	
			for (int j = 2; j < subfiles.size(); j++) {
				string ImagePath1 = string(databasePathString1) + "\\" + subfiles[j];
				Mat image = imread(ImagePath1);
				detector.detect(image, keypoints);
				Mat descriptors;
				descriptor.compute(image, keypoints, descriptors);
				trainer.add(descriptors);
				keypoints.clear();
			}
		
	}
	cv::Mat vocabulary = trainer.cluster();
	cv::FileStorage file("vocab.xml", cv::FileStorage::WRITE);
	file << "vocab" << vocabulary;
	file.release();

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();
	cv::BOWImgDescriptorExtractor dextract(extractor, matcher);
	dextract.setVocabulary(vocabulary);
	Mat alldescs;
	Mat alllabels_cannon;
	Mat alllabels_chair;
	Mat alllabels_crocodile;
	Mat alllabels_elephant;
	Mat alllabels_flamingo;
	Mat alllabels_helicopter;
	Mat alllabels_Motorbikes;
	Mat alllabels_scissors;
	Mat alllabels_strawberry;
	Mat alllabels_sunflower;
	for (int i = 2; i < files.size(); i++) {
		string imagePath = databasePathString + "\\" + files[i];
		const char* databasePathString1 = imagePath.c_str();
		vector<string> subfiles = getFiles(databasePathString1);
		for (int j = 2; j < subfiles.size(); j++) {
			string ImagePath1 = string(databasePathString1) + "\\" + subfiles[j];
			Mat image = imread(ImagePath1);
			
			detector.detect(image, keypoints);
			Mat descriptors;
			descriptor.compute(image, keypoints, descriptors);
			cv::Mat desc;
			dextract.compute(image, keypoints, desc);
			alldescs.push_back(desc);
			if (files[i] == "cannon")
				alllabels_cannon.push_back(0);
			else
				alllabels_cannon.push_back(1);
			
			if (files[i] == "chair")
				alllabels_chair.push_back(0);
			else
				alllabels_chair.push_back(1);
			
			if (files[i] == "crocodile")
				alllabels_crocodile.push_back(0);
			else
				alllabels_crocodile.push_back(1);
			if (files[i] == "elephant")
				alllabels_elephant.push_back(0);
			else
				alllabels_elephant.push_back(1);
			if (files[i] == "flamingo")
				alllabels_flamingo.push_back(0);
			else
				alllabels_flamingo.push_back(1);
			if (files[i] == "helicopter")
				alllabels_helicopter.push_back(0);
			else
				alllabels_helicopter.push_back(1);
			if (files[i] == "Motorbikes")
				alllabels_Motorbikes.push_back(0);
			else
				alllabels_Motorbikes.push_back(1);
			if (files[i] == "scissors")
				alllabels_scissors.push_back(0);
			else
				alllabels_scissors.push_back(1);
			if (files[i] == "strawberry")
				alllabels_strawberry.push_back(0);
			else
				alllabels_strawberry.push_back(1);
			if (files[i] == "sunflower")
				alllabels_sunflower.push_back(0);
			else
				alllabels_sunflower.push_back(1);
			    keypoints.clear();
		}
		
	}
	
	CvSVM svm_cannon;
	CvSVMParams params_cannon;
	svm_cannon.train_auto(alldescs, alllabels_cannon, Mat(), Mat(), params_cannon);
	svm_cannon.save("svm_cannon.xml");

	CvSVM svm_chair;
	CvSVMParams params_chair;
	svm_chair.train_auto(alldescs, alllabels_chair, Mat(), Mat(), params_chair);
	svm_chair.save("svm_chair.xml");

	CvSVM svm_crocodile;
	CvSVMParams params_crocodile;
	svm_crocodile.train_auto(alldescs, alllabels_crocodile, Mat(), Mat(), params_crocodile);
	svm_crocodile.save("svm_crocodile.xml");

	CvSVM svm_elephant;
	CvSVMParams params_elephant;
	svm_elephant.train_auto(alldescs, alllabels_elephant, Mat(), Mat(), params_elephant);
	svm_elephant.save("svm_elephant.xml");

	CvSVM svm_flamingo;
	CvSVMParams params_flamingo;
	svm_flamingo.train_auto(alldescs, alllabels_flamingo, Mat(), Mat(), params_flamingo);
	svm_flamingo.save("svm_flamingo.xml");

	CvSVM svm_helicopter;
	CvSVMParams params_helicopter;
	svm_helicopter.train_auto(alldescs, alllabels_helicopter, Mat(), Mat(), params_helicopter);
	svm_helicopter.save("svm_helicopter.xml");

	CvSVM svm_Motorbikes;
	CvSVMParams params_Motorbikes;
	svm_Motorbikes.train_auto(alldescs, alllabels_Motorbikes, Mat(), Mat(), params_Motorbikes);
	svm_Motorbikes.save("svm_Motorbikes.xml");

	CvSVM svm_scissors;
	CvSVMParams params_scissors;
	svm_scissors.train_auto(alldescs, alllabels_scissors, Mat(), Mat(), params_scissors);
	svm_scissors.save("svm_scissors.xml");

	CvSVM svm_strawberry;
	CvSVMParams params_strawberry;
	svm_strawberry.train_auto(alldescs, alllabels_strawberry, Mat(), Mat(), params_strawberry);
	svm_strawberry.save("svm_strawberry.xml");

	CvSVM svm_sunflower;
	CvSVMParams params_sunflower;
	svm_sunflower.train_auto(alldescs, alllabels_sunflower, Mat(), Mat(), params_sunflower);
	svm_sunflower.save("svm_sunflower.xml");
}


int main(int argc, char** argv) {
	//train("imagedb");

	Mat test_img = imread("imagedb_test\\sunflower\\image_0085.jpg");

	cv::Mat vocabulary;
	cv::FileStorage file("vocab.xml", cv::FileStorage::READ);
	file["vocab"] >> vocabulary;
	file.release();

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	SiftFeatureDetector detector = SiftFeatureDetector();
	cv::Ptr<cv::DescriptorExtractor> descriptor = new cv::SiftDescriptorExtractor();
	cv::BOWImgDescriptorExtractor dextract(descriptor, matcher);
	dextract.setVocabulary(vocabulary);

	vector<KeyPoint> keypoints;
	detector.detect(test_img, keypoints);
	Mat descriptors;
	descriptor->compute(test_img, keypoints, descriptors);
	cv::Mat desc;
	dextract.compute(test_img, keypoints, desc);

	CvSVM svm_cannon;
	svm_cannon.load("svm_cannon.xml");
	float prediction_cannon = svm_cannon.predict(desc, true);
	cout <<"cannon:"<< prediction_cannon<<endl;

	CvSVM svm_chair;
	svm_chair.load("svm_chair.xml");
	float prediction_chair = svm_chair.predict(desc, true);
	cout <<"chair:"<< prediction_chair<<endl;

	CvSVM svm_crocodile;
	svm_crocodile.load("svm_crocodile.xml");
	float prediction_crocodile = svm_crocodile.predict(desc, true);
	cout << "crocodile:"<<prediction_crocodile<<endl;

	CvSVM svm_elephant;
	svm_elephant.load("svm_elephant.xml");
	float prediction_elephant = svm_elephant.predict(desc, true);
	cout <<"elephant:"<< prediction_elephant<<endl;

	CvSVM svm_flamingo;
	svm_flamingo.load("svm_flamingo.xml");
	float prediction_flamingo = svm_flamingo.predict(desc, true);
	cout <<"flamingo:"<< prediction_flamingo<<endl;

	CvSVM svm_helicopter;
	svm_helicopter.load("svm_helicopter.xml");
	float prediction_helicopter = svm_helicopter.predict(desc, true);
	cout <<"helicopter:"<< prediction_helicopter<<endl;

	CvSVM svm_Motorbikes;
	svm_Motorbikes.load("svm_Motorbikes.xml");
	float prediction_Motorbikes = svm_Motorbikes.predict(desc, true);
	cout <<"motorbikes:"<< prediction_Motorbikes<<endl;

	CvSVM svm_scissors;
	svm_scissors.load("svm_scissors.xml");
	float prediction_scissors = svm_scissors.predict(desc, true);
	cout <<"scissors:"<< prediction_scissors<<endl;

	CvSVM svm_strawberry;
	svm_strawberry.load("svm_strawberry.xml");
	float prediction_strawberry = svm_strawberry.predict(desc, true);
	cout << "strawberry:"<<prediction_strawberry<<endl;

	CvSVM svm_sunflower;
	svm_sunflower.load("svm_sunflower.xml");
	float prediction_sunflower = svm_sunflower.predict(desc, true);
	cout <<"sunflower:"<< prediction_sunflower<<endl;

	float my_prediction[10];
	my_prediction[0] = prediction_cannon;
	my_prediction[1] = prediction_chair;
	my_prediction[2] = prediction_crocodile;
	my_prediction[3] = prediction_elephant;
	my_prediction[4] = prediction_flamingo;
	my_prediction[5] = prediction_helicopter;
	my_prediction[6] = prediction_Motorbikes;
	my_prediction[7] = prediction_scissors;
	my_prediction[8] = prediction_strawberry;
	my_prediction[9] = prediction_sunflower;
	float high_num = -2;
	for (int i = 0; i < 10; i++) {
		if (my_prediction[i] > high_num)
			high_num = my_prediction[i];
	}
	if (high_num == prediction_cannon) {
		cout << "My prediction is cannon with certainty:" << prediction_cannon<<endl;

	}
	if (high_num == prediction_chair) {
		cout << "My prediction is chair with certainty:" << prediction_chair << endl;
	}
	if (high_num == prediction_crocodile) {
		cout << "My prediction is crocodile with certainty:" << prediction_crocodile << endl;
	}
	if (high_num == prediction_elephant) {
		cout << "My prediction is elephant with certainty:" << prediction_elephant << endl;
	}
	if (high_num == prediction_flamingo) {
		cout << "My prediction is flamingo with certainty:" << prediction_flamingo << endl;
	}
	if (high_num == prediction_helicopter) {
		cout << "My prediction is helicopter with certainty:" << prediction_helicopter << endl;
	}
	if (high_num == prediction_Motorbikes) {
		cout << "My prediction is motorbike with certainty:" << prediction_Motorbikes << endl;
	}
	if (high_num == prediction_scissors) {
		cout << "My prediction is a pair of scissors with certainty:" << prediction_scissors << endl;
	}
	if (high_num == prediction_strawberry) {
		cout << "My prediction is starwberry with certainty:" << prediction_strawberry << endl;
	}
	if (high_num == prediction_sunflower) {
		cout << "My prediction is sunflower with certainty:" << prediction_sunflower << endl;
	}

	//
	
	return 0;
}
