#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMenuBar>
#include <QToolBar>
#include <QAction>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QStatusBar>
#include <QLabel>
#include <QGraphicsPixmapItem>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

private:
    QMenu *fileMenu;
    QAction *myFireAction;
    QAction *myMotionVectorAction;
    QAction *myOpenGLAction;

    QToolBar *fileToolBar;

private slots:
    void myFire();
    void myMotionVector();
    void myOpenGL();
    QStringList openDiaglog();
};
#endif // MAINWINDOW_H
