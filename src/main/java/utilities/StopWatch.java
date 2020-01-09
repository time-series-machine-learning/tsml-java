package utilities;

import java.io.Serializable;

public class StopWatch implements Serializable {
    private long timeStamp;
    private long time;
    private boolean paused = true;

    public StopWatch() {
        resetAndResume();
    }

    public StopWatch(boolean start) {
        this();
        if(start) {
            resume();
        }
    }

    public long lap() {
        if(!paused) {
            long nextTimeStamp = System.nanoTime();
            long diff = nextTimeStamp - this.timeStamp;
            time += diff;
            this.timeStamp = nextTimeStamp;
        }
        return time;
    }

    public boolean isPaused() {
        return paused;
    }

    public boolean isResumed() {
        return !paused;
    }

    public long getTimeNanos() {
        return time;
    }

    public StopWatch resume() {
        if(paused) {
            resumeAnyway();
            return this;
        } else {
            throw new IllegalStateException("already resumed");
        }
    }

    public StopWatch resumeAnyway() {
        paused = false;
        resetClock();
        return this;
    }

    public StopWatch pause() {
        if(!paused) {
            pauseAnyway();
            return this;
        } else {
            throw new IllegalStateException("already paused");
        }
    }

    public StopWatch pauseAnyway() { // doesn't matter if already paused
        paused = true;
        lap();
        return this;
    }

    public void resetClock() {
        timeStamp = System.nanoTime();
    }

    public void resetTime() {
        time = 0;
    }

    public void resetAndResume() {
        lap();
        resetTime();
        resetClock();
    }

    public void resetAndPause() {
        pauseAnyway();
        resetTime();
        resetClock();
    }

    public void add(long nanos) {
        time += nanos;
    }

    public void add(StopWatch stopWatch) {
        add(stopWatch.time);
    }
}
