package tsml.classifiers.distance_based.utils.system.timing;

import java.io.IOException;
import java.io.ObjectInputStream;

/**
 * Purpose: track time, ability to pause and add on time from another stop watch
 *
 * Contributors: goastler
 */
public class StopWatch extends Stated {
    private long timeStamp;
    private long elapsed;

    public StopWatch() {
        super();
    }
    
    public StopWatch(boolean start) {
        super(start);
    }

    private void writeObject(java.io.ObjectOutputStream stream)
            throws IOException {
        // record the current elapsed time, i.e. if state is currently started this laps the stopwatch
        if(isStarted()) {
            lap();
        }
        // then write the object
        stream.defaultWriteObject();
    }
    
    private void readObject(ObjectInputStream serialized) throws ClassNotFoundException, IOException
    {
        // read in the object
        serialized.defaultReadObject();
        // then reset the clock to current time so if the state is started the clock is resuming from now
        resetClock();
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
        // force the timer to update
        if(isStarted()) {
            lap();
        }
    }

    /**
     *
     * @return time in nanos
     */
    public long lap() {
        checkStarted();
        long nextTimeStamp = System.nanoTime();
        long diff = nextTimeStamp - this.timeStamp;
        elapsed += diff;
        this.timeStamp = nextTimeStamp;
        return elapsed;
    }
    
    public long getElapsedTimeStopped() {
        checkStopped();
        return elapsed;
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
    public void resetElapsedTime() {
        elapsed = 0;
    }

    /**
     * reset entirely
     */
    public void onReset() {
        resetElapsedTime();
        resetClock();
    }

    /**
     * add time from another source
     * @param nanos
     */
    public void add(long nanos) {
        elapsed += nanos;
    }

    public void add(StopWatch stopWatch) {
        add(stopWatch.elapsed);
    }

    @Override public String toString() {
        return "StopWatch{" +
            "time=" + elapsed +
            ", " + super.toString() +
            ", timeStamp=" + timeStamp +
            '}';
    }
}
