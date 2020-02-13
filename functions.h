#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <string>
#include <vector>
#include "ncnn/mat.h"

//读入图片，转成ncnn::Mat格式
ncnn::Mat imread(std::string imgPath);

//特征向量归一化
void normalize(std::vector<float> &feature);

//从标准格式数据集中读取图片和label
void loadStandardDataset(std::string folder, 
                         std::vector<ncnn::Mat> &imgs, 
                         std::vector<std::string> &labels);

//写入特征和对应的label到文件
void writeTrainSetFile(std::string filePath, 
                      const std::vector<std::vector<float>> &features, 
                      const std::vector<std::string> &labels);

//从文本文件读取图片路径
void readPathFromFile(std::string filePath, std::vector<std::string> &paths);

//写入特征到文件
void writeFeatures(std::string filePath, const std::vector<std::vector<float>> &features);

#endif // FUNCTIONS_H
