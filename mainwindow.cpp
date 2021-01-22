#include "fire.h"
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
    fileMenu = menuBar()->addMenu("&Open Video");
    myAction = new QAction("&MyAction", this);
    fileMenu->addAction(myAction);
    connect(myAction, SIGNAL(triggered(bool)), this, SLOT(myActionFn()));
    // fileToolBar->addAction(myAction);
    // connect(myAction, SIGNAL(triggered(bool)), this, SLOT(myActionFn()));
}

void MainWindow::myActionFn()
{
    // Find the logics of fire detection and print the report.
    QFileDialog dialog(this);
    dialog.setWindowTitle("Open Video");
    dialog.setFileMode(QFileDialog::AnyFile);
    // dialog.setNameFilter(tr("Images (*.png *.bmp *.jpg)"));
    QStringList filePaths;

    if (dialog.exec()) {
        filePaths = dialog.selectedFiles();
        // std::cout << filePaths.at(0).toUtf8().constData() << '\n';
        fire(filePaths.at(0));
    }
}


MainWindow::~MainWindow()
{
    delete ui;
}



