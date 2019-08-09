package utilities;

public class StopWatch {
    private long timeStamp;
    private long time;
    private boolean stopped = true;

    public StopWatch() {

    }

    public long lap() {
        if(stopped) {
            throw new IllegalStateException("cannot lap while stopped");
        }
        long nextTimeStamp = System.nanoTime();
        long diff = nextTimeStamp - this.timeStamp;
        time += diff;
        this.timeStamp = nextTimeStamp;
        return time;
    }

    public long getTimeNanos() {
        return time;
    }

    public void stop() {
        stopped = true;
        timeStamp = -1;
    }

    public void start() {
        timeStamp = System.nanoTime();
        stopped = false;
    }

    public void reset() {
        time = 0;
    }
}
