#ifndef FACEFEATUREEX_H
#define FACEFEATUREEX_H

#include "ncnn/mat.h"
#include "ncnn/net.h"

class faceFeatureEx
{
public:
    faceFeatureEx();
    
    //得到单张图片feature
    std::vector<float> getFeature(const ncnn::Mat &img);
    
    //得到多张图片features
    void getFeatures(const std::vector<ncnn::Mat> &imgs, 
                       std::vector<std::vector<float>> &features);
private:
    ncnn::Net m_net;
    int m_featureDim;
};

#endif // FACEFEATUREEX_H
