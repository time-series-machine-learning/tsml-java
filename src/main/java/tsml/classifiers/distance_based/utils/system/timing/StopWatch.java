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
    // total time excluding the current lap
    private long elapsedTime;
    private long lapTimeStamp;
    private long lapTime;

    public StopWatch() {
        super();
    }
    
    public StopWatch(boolean start) {
        super(start);
    }
    
    /**
     * The elapsed time since the stopwatch was last reset.
     * @return
     */
    public long elapsedTime() {
        long result = elapsedTime;
        if(isStarted()) {
            result += System.nanoTime() - timeStamp;
        }
        return result;
    }

    /**
     * Gets the lap time for the current lap. This is the time since start() or lap() was last called
     * @return
     */
    public long lapTime() {
        // if started
        if(isStarted()) {
            // then add on the difference since the lap started
            lapTime = System.nanoTime() - lapTimeStamp;
        }
        return lapTime;
    }
    
    /**
     * Perform a lap operation. This provides the time since the last lap call or the elapsed time so far if lap has not been called yet. (Think F1 track recording car passing start line every lap, this reports the time taken to do 1 lap of the track)
     * @return the lap time
     */
    public long lap() {
        if(isStarted()) {
            // track the time stamp of the current lap
            long previousStartLapTimeStamp = lapTimeStamp;
            // build a new start time stamp for the new lap
            lapTimeStamp = System.nanoTime();
            // the difference between the time stamps is the lap time
            lapTime = lapTimeStamp - previousStartLapTimeStamp;
        }
        return lapTime;
    }

    /**
     * The timeStamp of when the current lap began.
     * @return
     */
    public long lapTimeStamp() {
        return lapTimeStamp;
    }

    /**
     * Timestamp of when the stopwatch began / was started.
     * @return
     */
    public long timeStamp() {
        return timeStamp;
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

    @Override public void start() {
        super.start();
        // update the clock to current time
        resetClock();
    }

    @Override public void stop() {
        // force the timer to update
        lap();
        // add on the diff between when the stopwatch was started and the timestamp given from the new lap, i.e. the stop timestamp
        elapsedTime += lapTimeStamp - timeStamp;
        super.stop();
    }
    
    public long lapAndStop() {
        stop();
        return lapTime;
    }
    
    public long optionalLap() {
        if(isStarted()) {
            lap();
        }
        return lapTime;
    }
    
    public long elapsedTimeStopped() {
        checkStopped();
        return elapsedTime();
    }
    
    public long elapsedTimeStarted() {
        checkStarted();
        return elapsedTime();
    }
    
    public long lapTimeStarted() {
        checkStarted();
        return lapTime();
    }
    
    public long lapTimeStopped() {
        checkStopped();
        return lapTime();
    }

    /**
     * reset the clock, useful post serialisation
     */
    public void resetClock() {
        timeStamp = System.nanoTime();
        lapTimeStamp = timeStamp;
    }

    /**
     * reset time count
     */
    public void resetElapsed() {
        elapsedTime = 0;
        lapTime = 0;
    }

    @Override public void reset() {
        super.reset();
        resetElapsed();
        resetClock();
    }

    /**
     * add time from another source
     * @param nanos
     */
    public void add(long nanos) {
        elapsedTime += nanos;
    }

    public void add(StopWatch stopWatch) {
        add(stopWatch.elapsedTime());
    }

    @Override public String toString() {
        return "StopWatch{" +
            "elapsedTime=" + elapsedTime() +
            ", lapTime=" + lapTime() +
            ", " + super.toString() +
            ", timeStamp=" + timeStamp() +
            ", lapTimeStamp=" + lapTimeStamp() +
            '}';
    }
}
