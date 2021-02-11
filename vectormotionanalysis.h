#ifndef VECTORMOTIONANALYSIS_H
#define VECTORMOTIONANALYSIS_H

#include <iostream>
#include <QApplication>
#include <cstring>


class vectorMotionAnalysis
{
public:
    vectorMotionAnalysis(QString fileName);
    void vectorMotionAnalysisLucas(QString fileName);
    void vectorMotionAnalysisFarneback(QString fileName);
};

#endif // VECTORMOTIONANALYSIS_H
