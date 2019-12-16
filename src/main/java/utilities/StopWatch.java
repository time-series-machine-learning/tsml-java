package utilities;

import java.io.Serializable;

public class StopWatch implements Serializable {
    private long timeStamp;
    private long time;
    private boolean paused = true;

    public StopWatch() {
        reset();
    }

    public StopWatch(boolean start) {
        this();
        if(start) {
            resume();
        }
    }

    public long lap() {
        long nextTimeStamp = System.nanoTime();
        long diff = nextTimeStamp - this.timeStamp;
        time += diff;
        this.timeStamp = nextTimeStamp;
        return time;
    }

    public long getTimeNanos() {
        return time;
    }

    public void resume() {
        if(paused) {
            paused = false;
            resetClock();
        }
    }

    public long pause() {
        if(!paused) {
            paused = true;
            lap();
        }
        return getTimeNanos();
    }

    public void resetClock() {
        timeStamp = System.nanoTime();
    }

    public void resetTime() {
        time = 0;
    }

    public void reset() {
        pause();
        resume();
        resetTime();
    }

    public void add(long nanos) {
        time += nanos;
    }
}
