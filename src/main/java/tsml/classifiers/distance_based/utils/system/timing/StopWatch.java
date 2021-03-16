/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.utils.system.timing;

import tsml.classifiers.distance_based.utils.experiment.TimeSpan;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.time.Duration;

/**
 * Purpose: track time, ability to pause and add on time from another stop watch
 *
 * Contributors: goastler
 */
public class StopWatch extends Stated {
    // track the last time the elapsedTime was updated
    private long lastUpdateTimeStamp;
    // the cumulative elapsed time
    private long elapsedTime;

    public StopWatch() {
        super(false);
    }
    
    public StopWatch(boolean start) {
        super(start);
    }
    
    public StopWatch(long startTimeStamp) {
        super(false);
        start(startTimeStamp);
    }

    /**
     * Get the elapsed time. If the StopWatch is started, this will update the elapsedTime with the difference since start() or this method were last called.
     * @return
     */
    public long elapsedTime() {
        return elapsedTime(System.nanoTime());
    }

    /**
     * Update and get the elapsed time. If the stopwatch is started, this will update the elapsed time with the difference between the given time stamp and the last recorded timestamp (either the start time stamp or the time stamp from the most recent call to this method).
     * @param timeStamp
     * @return
     */
    public long elapsedTime(long timeStamp) {
        if(isStarted()) {
            if(lastUpdateTimeStamp > timeStamp) {
                throw new IllegalStateException("last update time stamp is from the future: " + lastUpdateTimeStamp + " > " + timeStamp);
            }
            final long diff = timeStamp - lastUpdateTimeStamp;
            elapsedTime += diff;
            lastUpdateTimeStamp = timeStamp;
        }
        return elapsedTime;
    }

    /**
     * Start the StopWatch at the current time.
     */
    public void start() {
        start(System.nanoTime());
    }

    /**
     * Start the StopWatch from the given time.
     * @param startTimeStamp
     */
    public void start(long startTimeStamp) {
        super.start();
        setStartTimeStamp(startTimeStamp);
    }
    
    public void stop(long timeStamp) {
        elapsedTime(timeStamp);
        super.stop();
    }
    
    public void stop() {
        stop(System.nanoTime());
    }

    /**
     * Set the start time irrelevant of current state.
     * @param startTimeStamp
     */
    private void setStartTimeStamp(long startTimeStamp) {
        if(startTimeStamp > System.nanoTime()) {
            throw new IllegalArgumentException("cannot set start time in the future");
        }
        lastUpdateTimeStamp = startTimeStamp;
    }

    /**
     * Set the elapsed time.
     * @param elapsedTime
     */
    public void setElapsedTime(long elapsedTime) {
        if(elapsedTime < 0) {
            throw new IllegalArgumentException("elapsed time cannot be less than 0");
        }
        this.elapsedTime = elapsedTime;
    }

    public void resetElapsedTime() {
        setElapsedTime(0);
    }
    
    /**
     * Reset the elapsed time to zero and invalidate the start time.
     */
    public void reset() {
        optionalStop();
        resetElapsedTime();
        setStartTimeStamp(System.nanoTime());
    }    
    
    /**
     * add time to the elapsed time
     * @param nanos
     */
    public void add(long nanos) {
        elapsedTime += nanos;
    }
    
    public void add(long startTimeStamp, long stopTimeStamp) {
        if(stopTimeStamp < startTimeStamp) {
            throw new IllegalArgumentException("start before stop");
        }
        add(stopTimeStamp - startTimeStamp);
    }

    @Override public String toString() {
        return "StopWatch{" +
            "elapsedTime=" + elapsedTime() +
            ", " + super.toString() +
            '}';
    }

    /**
     * Get the time stamp when the elapsed time was last updated.
     * @return
     */
    public long timeStamp() {
        return lastUpdateTimeStamp;
    }

    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
        ois.defaultReadObject();
        // any stopwatch read from file should begin in a stopped state
        setStartTimeStamp(0);
        if(isStarted()) {
            super.stop();
        }
    }
    
    private void writeObject(ObjectOutputStream oos) throws ClassNotFoundException, IOException {
        // update the elapsed time
        elapsedTime();
        oos.defaultWriteObject();
    }
    
    public TimeSpan toTimeSpan() {
        return new TimeSpan(elapsedTime());
    }
}
