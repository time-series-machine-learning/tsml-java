package utilities;

public class StopWatch {
    private long timeStamp;
    private long time;

    public StopWatch() {

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

    public void resetClock() {
        timeStamp = System.nanoTime();
    }

    public void resetTime() {
        time = 0;
    }

    public void reset() {
        resetTime();
        resetClock();
    }

    public void add(long nanos) {
        time += nanos;
    }
}
