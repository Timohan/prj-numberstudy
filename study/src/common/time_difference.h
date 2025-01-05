/*!
 * \file
 * \brief file time_difference.h
 *
 * time difference class
 *
 * Copyright of Timo Hannukkala. All rights reserved.
 *
 * \author Timo Hannukkala <timohannukkala@hotmail.com>
 */
#ifndef TIMEDIFFERENCE_H
#define TIMEDIFFERENCE_H


#include <time.h>

class TimeDifference
{
public:
public:
    TimeDifference();

    double elapsedTimeFromBegin() const;
    void resetTimer(double addTime = 0);
    double operator-(const TimeDifference &diff);
    timespec getBeginTime() const;
    void pause();
    void play();

private:
    timespec beginTime;
    timespec pauseBeginTime;
    double m_addTime;
    bool m_paused;
};

#endif //TIMEDIFFERENCE_H
