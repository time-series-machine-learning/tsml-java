package tsml.classifiers.distance_based.proximity;

import tsml.classifiers.distance_based.utils.stopwatch.StopWatch;

/**
 * Purpose: helper functions to extend contracting
 * <p>
 * Contributors: goastler
 */
public class TimeContracter {

    public TimeContracter(final StopWatch timer) {
        this.timer = timer;
    }

    public TimeContracter() {
        this(new StopWatch());
    }

    public long getRemainingTrainTime() {
        if(!hasTimeLimit()) {
            throw new IllegalStateException("time limit not set");
        }
        long timeTaken = timer.getTimeNanos();
        long diff = timeLimit - timeTaken;
        return Math.max(0, diff);
    }

    public boolean hasRemainingTrainTime() {
        return getRemainingTrainTime() > 0;
    }

    public boolean hasTimeLimit() {
        return timeLimit > 0;
    }

    private long timeLimit = 0;
    private final StopWatch timer;

    public void setTimeLimit(final long time) {
        timeLimit = time;
    }

    public long getTimeLimit() {
        return timeLimit;
    }

    public StopWatch getTimer() {
        return timer;
    }
}
