/*!
 * \file
 * \brief file time_difference.cpp
 *
 * time difference class
 *
 * Copyright of Timo Hannukkala. All rights reserved.
 *
 * \author Timo Hannukkala <timohannukkala@hotmail.com>
 */

#include "time_difference.h"

TimeDifference::TimeDifference() :
    m_addTime(0),
    m_paused(false)
{
    clock_gettime(CLOCK_MONOTONIC, &beginTime);
}

/*!
 * \brief TimeDifference::elapsedTimeFromBegin
 *
 * elapsed (running) time
 *
 * \return elapsed time from the beginning
 */
double TimeDifference::elapsedTimeFromBegin() const
{
    timespec currentTime;
    clock_gettime(CLOCK_MONOTONIC, &currentTime);
    if (!m_paused) {
        return m_addTime + static_cast<double>(currentTime.tv_sec) - static_cast<double>(beginTime.tv_sec) + static_cast<double>(currentTime.tv_nsec - beginTime.tv_nsec)/1000000000.0;
    }
    return m_addTime + static_cast<double>(currentTime.tv_sec - beginTime.tv_sec) + static_cast<double>(currentTime.tv_nsec - beginTime.tv_nsec)/1000000000.0
    - (static_cast<double>(currentTime.tv_sec - pauseBeginTime.tv_sec) + static_cast<double>(currentTime.tv_nsec - pauseBeginTime.tv_nsec)/1000000000.0);
}

/*!
 * \brief TimeDifference::resetTimer
 *
 * reset timer and sets it running (play)
 *
 * \param addTime
 */
void TimeDifference::resetTimer(double addTime)
{
    clock_gettime(CLOCK_MONOTONIC, &beginTime);
    m_addTime = addTime;
    m_paused = false;
}

/*!
 * \brief TimeDifference::pause
 *
 * sets this time different "pause" from running (play)
 */
void TimeDifference::pause()
{
    if (m_paused) {
        return;
    }
    m_paused = true;
    clock_gettime(CLOCK_MONOTONIC, &pauseBeginTime);
}

/*!
 * \brief TimeDifference::play
 *
 * sets this time different "run" from pause
 */
void TimeDifference::play()
{
    if (!m_paused) {
        return;
    }
    resetTimer(elapsedTimeFromBegin());
}

/*!
 * \brief TimeDifference::getBeginTime
 *
 * get time begin
 */
timespec TimeDifference::getBeginTime() const
{
    return beginTime;
}

/*!
 * \brief TimeDifference::operator-
 *
 * minus for time
 */
double TimeDifference::operator-(const TimeDifference &diff)
{
    return elapsedTimeFromBegin() - diff.elapsedTimeFromBegin();
}

