#include <iostream>
#include <dirent.h>
#include <sys/stat.h>

#include "faceFeatureEx.h"
#include "functions.h"

int main(int argc, char **argv)
{
    if(argc != 3){
        std::cout<<"参数错误！"<<std::endl;
        exit(1);
    }
    
    //输入路径（文件夹或文本文件）
    std::string inputPath = argv[1];
    //输出文件
    std::string outputFile = argv[2];
    
    //区别输入路径是文件还是文件夹
    struct stat s_buf;
    stat(inputPath.data(),&s_buf);
    if(S_ISDIR(s_buf.st_mode)){
        //加载数据集
        std::vector<ncnn::Mat> imgs;
        std::vector<std::string> labels;
        loadStandardDataset(inputPath,imgs,labels);
        
        //提取特征
        std::vector<std::vector<float>> features;
        faceFeatureEx extractor;
        extractor.getFeatures(imgs,features);
        
        //写入到文件
        writeTrainSetFile(outputFile,features,labels);
    }
    else{
        //从输入文件读取图片路径
        std::vector<std::string> paths;
        readPathFromFile(inputPath,paths);
        
        //从图片路径读取图片
        std::vector<ncnn::Mat> imgs;
        for(int i=0;i<paths.size();i++){
            imgs.push_back(imread(paths[i]));
        }
        
        //提取特征
        std::vector<std::vector<float>> features;
        faceFeatureEx extractor;
        extractor.getFeatures(imgs,features);
        
        //写入文件
        writeFeatures(outputFile,features);
    }
    
    return 0;
}
