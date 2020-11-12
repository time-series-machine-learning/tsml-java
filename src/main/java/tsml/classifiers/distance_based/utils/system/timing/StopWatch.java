package tsml.classifiers.distance_based.utils.system.timing;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * Purpose: track time, ability to pause and add on time from another stop watch
 *
 * Contributors: goastler
 */
public class StopWatch extends Stated {
    private transient long timeStamp;
    private long elapsed;
    private long diff;

    public StopWatch() {
        super();
    }
    
    public StopWatch(boolean start) {
        super(start);
    }

    private void writeObject(ObjectOutputStream stream)
            throws IOException {
        // record the current elapsed time, i.e. if state is currently started this laps the stopwatch
        if(isStarted()) {
            lap();
        }
        // then write the object
        stream.defaultWriteObject();
        // NOTE does not change the state of this stopwatch so will keep running if started!
    }
    
    private void readObject(ObjectInputStream serialized) throws ClassNotFoundException, IOException
    {
        // read in the object
        serialized.defaultReadObject();
        // then reset the clock to current time so if the state is started the clock is resuming from now
        resetClock();
    }

    /**
     * Get the previous elapsed time, i.e. the time stamp of the second most recent lap() or split() call.
     * @return
     */
    public long getPreviousElapsedTime() {
        if(!isStarted()) {
            throw new IllegalStateException("not started");
        }
        return timeStamp;
    }

    @Override public void start() {
        super.start();
        // update the clock to current time
        resetClock();
    }

    @Override public void stop() {
        super.stop();
        // force the timer to update
        if(isStarted()) {
            lap();
        }
    }

    /**
     * Perform a lap operation. This returns the total elapsed time at the time of lapping (think F1 track recording car passing start line every lap)
     * @return total elapsed time in nanos
     */
    public long lap() {
        split();
        return getElapsedTime();
    }

    /**
     * Perform a split operation. This returns the elapsed time since the last lap or split operation, i.e. the diff.
     * @return DIFFERENCE in time in nanos since last split / lap call
     */
    public long split() {
        checkStarted();
        long nextTimeStamp = System.nanoTime();
        diff = nextTimeStamp - this.timeStamp;
        elapsed += diff;
        this.timeStamp = nextTimeStamp;
        return getSplitTime();
    }

    public long splitAndStop() {
        stop();
        return getElapsedTimeStopped();
    }
    
    public long lapAndStop() {
        stop();
        return getSplitTimeStopped();
    }
    
    public long lapIfStarted() {
        if(isStarted()) {
            lap();
        }
        return getElapsedTime();
    }
    
    public long splitIfStarted() {
        if(isStarted()) {
            split();
        }
        return getSplitTime();
    }
    
    public long getElapsedTimeStopped() {
        checkStopped();
        return getElapsedTime();
    }

    /**
     * NOTE this does not update the time! Get the elapsed time as calculated in the last split or lap call. This will be exactly the same value as returned by the most recent lap call.
     * @return
     */
    public long getElapsedTime() {
        return elapsed;
    }
    
    public long getSplitTimeStopped() {
        checkStopped();
        return getSplitTime();
    }
    
    /**
     * NOTE this does not update the time! Get the split time as calculated in the last split or lap call. This will be exactly the same value as returned by the most recent split call.
     * @return
     */
    public long getSplitTime() {
        return diff;
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
     * reset time count
     */
    public void resetSplitTime() {
        diff = 0;
    }

    /**
     * reset entirely
     */
    public void onReset() {
        resetElapsedTime();
        resetSplitTime();
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
        add(stopWatch.lapIfStarted());
    }

    @Override public String toString() {
        return "StopWatch{" +
            "time=" + elapsed +
            ", " + super.toString() +
            ", timeStamp=" + timeStamp +
            '}';
    }
}
