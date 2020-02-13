#include "functions.h"
#include "math.h"

#include "CImg.h"
#include <dirent.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>

void CImg2NcnnImg(const cil::CImg<unsigned char> &cimg, ncnn::Mat &ncnnImg)
{
    int w = cimg.width();
    int h = cimg.height();
    ncnnImg.create(w,h,3);
    
    for(int y=0;y<h;y++){
        for(int x=0;x<w;x++){
            for(int c=0;c<3;c++){
                ncnnImg.channel(c).row(y)[x] = cimg.atX(x,y,0,c);
            }
        }
    }
}

ncnn::Mat imread(std::string imgPath)
{
    //用cimg读取图像
    cil::CImg<unsigned char> cimg(imgPath.c_str());
    if(cimg.empty()){
        std::cout<<"读取图片"<<imgPath<<"失败！"<<std::endl;
        exit(1);
    }
    
    //转换为ncnn图像格式
    ncnn::Mat ncnnImg;
    CImg2NcnnImg(cimg,ncnnImg);
    
    return ncnnImg;
}

void normalize(std::vector<float> &feature)
{
    float sum = 0;
    for (auto it = feature.begin(); it != feature.end(); it++)
        sum += (float)*it * (float)*it;
    sum = std::sqrt(sum);
    for (auto it = feature.begin(); it != feature.end(); it++)
        *it /= sum;
}

//遍历一个目录
void traverseFile(const std::string directory, std::vector<std::string> &files)
{
    std::string prefix = directory;
    if(directory[directory.length()-1] != '/')
        prefix += '/';
    
    files.clear();
    
    const char * char_dir = directory.data();
    
    DIR* dir = opendir(char_dir);//打开指定目录
    dirent* p = nullptr;//定义遍历指针
    while((p = readdir(dir)) != nullptr)//开始逐个遍历
    {
        //linux平台下目录中有"."和".."隐藏文件，需要过滤掉
        if(p->d_name[0] != '.')//d_name是一个char数组，存放当前遍历到的文件名
        {
            std::string name = prefix + std::string(p->d_name);
            files.push_back(name);
        }
    }
    closedir(dir);//关闭指定目录
}

void loadStandardDataset(std::string folder, 
                         std::vector<ncnn::Mat> &imgs, 
                         std::vector<std::string> &labels)
{
    std::vector<std::string> main_files;
    traverseFile(folder,main_files);
    
    std::vector<std::string> main_dirs;
    for(size_t i=0;i<main_files.size();i++){
        //判断是否是文件夹,不是就跳过
        char const*path = main_files[i].data();
        struct stat s_buf;
        stat(path,&s_buf);
        if(!S_ISDIR(s_buf.st_mode))
            continue;
        
        main_dirs.push_back(main_files[i]);
    }
    
    for(size_t i=0;i<main_dirs.size();i++){
        std::vector<std::string> sub_files;
        traverseFile(main_dirs[i],sub_files);
        
        std::string label = main_dirs[i].substr(main_dirs[i].find_last_of('/')+1,main_dirs[i].length()-1);
        
        for(int j=0;j<sub_files.size();j++){
            //用cimg读取图像
            cil::CImg<unsigned char> cimg(sub_files[j].c_str());
            if(cimg.empty())
                continue;
            
            //转换为ncnn图像格式
            ncnn::Mat ncnnImg;
            CImg2NcnnImg(cimg,ncnnImg);
            
            imgs.push_back(ncnnImg);
            labels.push_back(label);
        }
    }
}

void writeTrainSetFile(std::string filePath, 
                      const std::vector<std::vector<float>> &features, 
                      const std::vector<std::string> &labels)
{
    std::string contentStr = "";
    for(int i=0;i<labels.size();i++){
        std::string line = labels[i] + ":";
        for(int j=0;j<features[i].size();j++){
            std::string fstr = std::to_string(features[i][j]);
            fstr = fstr.substr(0,fstr.find_first_of(".")+3);
            line += fstr + ",";
        }
        line[line.length()-1] = '\n';
        
        contentStr += line;
    }
    contentStr = contentStr.substr(0,contentStr.length()-1);
    
    std::ofstream ofs(filePath,std::ios::out);
    if(!ofs.is_open()){
        std::cout<<"文件\""<<filePath<<"\"打开失败!"<<std::endl;
        exit(1);
    }
    ofs << contentStr;
    
    ofs.close();
}

void readPathFromFile(std::string filePath, std::vector<std::string> &paths)
{
    std::ifstream ifs(filePath,std::ios::in);
    if(!ifs.is_open()){
        std::cout<<"文件\""<<filePath<<"\"打开失败!"<<std::endl;
        exit(1);
    }
    
    std::string line;
    while(getline(ifs,line)){
        //去除换行符
        if(line[line.length()-1] == '\n'){
            line = line.substr(0,line.length()-1);
        }
        
        paths.push_back(line);
    }
    
    ifs.close();
}

void writeFeatures(std::string filePath, const std::vector<std::vector<float>> &features)
{
    std::string contentStr = "";
    for(int i=0;i<features.size();i++){
        std::string line = "";
        for(int j=0;j<features[i].size();j++){
            std::string fstr = std::to_string(features[i][j]);
            fstr = fstr.substr(0,fstr.find_first_of(".")+3);
            line += fstr + ",";
        }
        line[line.length()-1] = '\n';
        
        contentStr += line;
    }
    contentStr = contentStr.substr(0,contentStr.length()-1);
    
    std::ofstream ofs(filePath,std::ios::out);
    if(!ofs.is_open()){
        std::cout<<"文件\""<<filePath<<"\"打开失败!"<<std::endl;
        exit(1);
    }
    ofs << contentStr;
    
    ofs.close();
}
