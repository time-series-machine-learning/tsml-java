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

/**
 * Purpose: track time, ability to pause and add on time from another stop watch
 *
 * Contributors: goastler
 */
public class StopWatch extends Stated {
    private transient long timeStamp;
    private long time;

    public StopWatch() {
        reset();
    }

    public long getStartTime() {
        if(!isStarted()) {
            throw new IllegalStateException("not started");
        }
        return timeStamp;
    }

    @Override
    protected void beforeStart() {
        super.beforeStart();
        timeStamp = System.nanoTime();
    }

    @Override
    protected void beforeStop() {
        super.beforeStop();
        lap();
    }

    /**
     * just update the time
     * @return
     */
    public long lap() {
        if(isStarted()) {
            long nextTimeStamp = System.nanoTime();
            long diff = nextTimeStamp - this.timeStamp;
            time += diff;
            this.timeStamp = nextTimeStamp;
            return time;
        } else {
            throw new IllegalStateException("not started, cannot lap");
        }
    }

    /**
     *
     * @return time in nanos
     */
    public long getTime() {
        return time;
    }

    /**
     * reset the clock, useful post serialisation
     */
    public void resetClock() {
        timeStamp = System.nanoTime();
    }

    /**
     * reset time count
     */
    public void resetTime() {
        time = 0;
    }

    /**
     * reset entirely
     */
    public void onReset() {
        resetTime();
        resetClock();
    }

    /**
     * add time from another source
     * @param nanos
     */
    public void add(long nanos) {
        time += nanos;
    }

    public void add(StopWatch stopWatch) {
        add(stopWatch.time);
    }

    @Override public String toString() {
        return "StopWatch{" +
            "time=" + time +
            ", " + super.toString() +
            ", timeStamp=" + timeStamp +
            '}';
    }
}
