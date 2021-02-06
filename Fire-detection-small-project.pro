QT       += core gui
QT       += opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

TARGET = opencvtest
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    fire.cpp \
    main.cpp \
    mainwindow.cpp \
    opengl.cpp \
    vectormotionanalysis.cpp

HEADERS += \
    fire.h \
    mainwindow.h \
    opengl.h \
    vectormotionanalysis.h

FORMS += \
    mainwindow.ui

INCLUDEPATH += D:\opencv_qt\opencv\build\include

LIBS += D:\opencv_qt\opencv_build\bin\libopencv_core451.dll
LIBS += D:\opencv_qt\opencv_build\bin\libopencv_highgui451.dll
LIBS += D:\opencv_qt\opencv_build\bin\libopencv_imgcodecs451.dll
LIBS += D:\opencv_qt\opencv_build\bin\libopencv_imgproc451.dll
LIBS += D:\opencv_qt\opencv_build\bin\libopencv_features2d451.dll
LIBS += D:\opencv_qt\opencv_build\bin\libopencv_calib3d451.dll
LIBS += D:\opencv_qt\opencv_build\bin\libopencv_video451.dll
LIBS += D:\opencv_qt\opencv_build\bin\libopencv_videoio451.dll

# more correct variant, how set includepath and libs for mingw
# add system variable: OPENCV_SDK_DIR=D:/opencv/opencv-build/install
# read http://doc.qt.io/qt-5/qmake-variable-reference.html#libs

# Default rules for deployment.
#qnx: target.path = /tmp/$${TARGET}/bin
#else: unix:!android: target.path = /opt/$${TARGET}/bin
#!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    test_fire_1.mp4 \
    test_fire_2.mp4
