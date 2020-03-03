package tsml.classifiers.distance_based.utils;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

/**
 * Purpose: track time, ability to pause and add on time from another stop watch
 *
 * Contributors: goastler
 */
public class StopWatch extends Stated implements Serializable {
    private transient long timeStamp;
    private long time;
    private transient Set<StopWatch> listeners = new HashSet<>();

    public StopWatch() {
        super();
    }

    public StopWatch(State state) {
        super(state);
    }

    /**
     * listen to another stopwatch, updating the other stopwatch's state when this stopwatch's state changes.
     * @param other
     */
    public void addListener(StopWatch other) {
        super.addListener(other);
        listeners.add(other);
    }

    public void removeListener(StopWatch other) {
        super.removeListener(other);
        listeners.remove(other);
    }

    /**
     * just update the time
     * @return
     */
    public long lap() {
        if(isEnabled()) {
            uncheckedLap();
        }
        return time;
    }

    /**
     * this is dangerous, don't use unless you know what you're doing. It updates the time without any checks on the
     * state of the stopwatch.
     * @return
     */
    private long uncheckedLap() {
        long nextTimeStamp = System.nanoTime();
        long diff = nextTimeStamp - this.timeStamp;
        time += diff;
        this.timeStamp = nextTimeStamp;
        return time;
    }

    public long getTimeNanos() {
        return time;
    }

    /**
     * enable irrelevant of current state
     * @return
     */
    @Override public boolean enableAnyway() {
        boolean change = super.enableAnyway();
        if(change) {
            resetClock();
        }
        return change;
    }

    /**
     * disable irrelevant of current state
     * @return
     */
    @Override public boolean disableAnyway() {
        boolean change = super.disableAnyway();
        if(change) {
            uncheckedLap();
        }
        return change;
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
    public void reset() {
        resetAndDisable();
    }

    public void resetAndEnable() {
        disableAnyway();
        resetTime();
        enable();
    }

    public void resetAndDisable() {
        disableAnyway();
        resetTime();
        resetClock();
    }

    /**
     * add time from another source
     * @param nanos
     */
    public void add(long nanos) {
        time += nanos;
        for(StopWatch other : listeners) {
            other.add(nanos);
        }
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
