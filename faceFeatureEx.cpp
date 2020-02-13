#include "faceFeatureEx.h"
#include "functions.h"

faceFeatureEx::faceFeatureEx()
{
    std::string paramFile = "./model/MobileFaceNet.param";
    std::string binFile = "./model/MobileFaceNet.bin";
    m_featureDim = 128;
    
    m_net.load_param(paramFile.c_str());
    m_net.load_model(binFile.c_str());
}

std::vector<float> faceFeatureEx::getFeature(const ncnn::Mat &img)
{
    std::vector<float> feature;
    ncnn::Mat in;
    ncnn::resize_bilinear(img,in,112,112);
    
    ncnn::Extractor ex = m_net.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("fc1", out);
    feature.resize(m_featureDim);
    for (int i = 0; i < m_featureDim; i++)
        feature[i] = out[i];
    normalize(feature);
    return feature;
}

void faceFeatureEx::getFeatures(const std::vector<ncnn::Mat> &imgs, 
                                std::vector<std::vector<float> > &features)
{
    features.resize(imgs.size());
    for(int i=0;i<imgs.size();i++){
        features[i] = getFeature(imgs[i]);
    }
}
