package utilities;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

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

    public void addListener(StopWatch other) {
        super.addListener(other);
        listeners.add(other);
    }

    public void removeListener(StopWatch other) {
        super.removeListener(other);
        listeners.remove(other);
    }

    public long lap() {
        if(isEnabled()) {
            uncheckedLap();
        }
        return time;
    }

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

    @Override public boolean enableAnyway() {
        boolean change = super.enableAnyway();
        if(change) {
            resetClock();
        }
        return change;
    }

    @Override public boolean disableAnyway() {
        boolean change = super.disableAnyway();
        if(change) {
            uncheckedLap();
        }
        return change;
    }

    public void resetClock() {
        timeStamp = System.nanoTime();
    }

    public void resetTime() {
        time = 0;
    }

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
