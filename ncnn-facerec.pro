TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

HEADERS += \
  faceFeatureEx.h \
  functions.h

SOURCES += \
        faceFeatureEx.cpp \
        functions.cpp \
        main.cpp

INCLUDEPATH += /home/liu/libraries/cimg
INCLUDEPATH += /home/liu/libraries/ncnn/include
LIBS += /home/liu/libraries/ncnn/lib/libncnn.a
LIBS += -lgomp -lpthread
