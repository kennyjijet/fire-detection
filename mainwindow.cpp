#include "fire.h"
#include "opengl.h"
#include "vectormotionanalysis.h"
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <iostream>

cv::Mat	pixelMotion();
cv::Mat fireRGB(cv::Mat frame);

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    fileMenu = menuBar()->addMenu("&My Menu");
    myFireAction = new QAction("&Open Video", this);
    fileMenu->addAction(myFireAction);

    myMotionVectorAction = new QAction("&Motion Vector Analysis", this);
    fileMenu->addAction(myMotionVectorAction);

    myOpenGLAction = new QAction("&OpenGL", this);
    fileMenu->addAction(myOpenGLAction);

    connect(myFireAction, SIGNAL(triggered(bool)), this, SLOT(myFire()));
    connect(myMotionVectorAction, SIGNAL(triggered(bool)), this, SLOT(myMotionVector()));
    connect(myOpenGLAction, SIGNAL(triggered(bool)), this, SLOT(myOpenGL()));

}

void MainWindow::myFire()
{
    // Find the logics of fire detection and print the report.
    QFileDialog dialog(this);
    dialog.setWindowTitle("Open Video");
    dialog.setFileMode(QFileDialog::AnyFile);
    QStringList filePaths;

    if (dialog.exec()) {
        filePaths = dialog.selectedFiles();
        fire(filePaths.at(0));
    }
}

void MainWindow::myMotionVector()
{
    QFileDialog dialog(this);
    dialog.setWindowTitle("Open Video");
    dialog.setFileMode(QFileDialog::AnyFile);
    QStringList filePaths;

    if (dialog.exec()) {
        filePaths = dialog.selectedFiles();

        vectorMotionAnalysis();
    }

}

void MainWindow::myOpenGL()
{
    openGL();
}


MainWindow::~MainWindow()
{
    delete ui;
}



