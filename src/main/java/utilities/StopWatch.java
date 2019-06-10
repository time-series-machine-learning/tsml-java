package utilities;

public class StopWatch {
    private long lastTimestamp = -1;
    private long timeTakenNanos = 0;

    public StopWatch() {
        reset();
    }

    public long lapAndGet() {
        lap();
        return get();
    }

    public long get() {
        return timeTakenNanos;
    }

    public long lap() {
        long timestamp = System.nanoTime();
        long diff = timestamp - lastTimestamp;
        timeTakenNanos += diff;
        lastTimestamp = timestamp;
        return diff;
    }

    public void resetClock() {
        lastTimestamp = System.nanoTime();
    }

    public void reset() {
        timeTakenNanos = 0;
        resetClock();
    }
}
