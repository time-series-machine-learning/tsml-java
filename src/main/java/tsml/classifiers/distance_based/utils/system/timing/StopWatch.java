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
