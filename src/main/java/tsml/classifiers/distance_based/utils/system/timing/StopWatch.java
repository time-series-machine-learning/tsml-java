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
    private transient long startTimeStamp;
    // total time excluding the current lap
    private long elapsedTime;
    private long startLapTimeStamp;
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
        return elapsedTimeBeforeLap() + lapTime();
    }

    /**
     * The elapsed time when lap was last called.
     * @return
     */
    public long elapsedTimeBeforeLap() {
        return elapsedTime;
    }

    /**
     * Gets the lap time for the current lap. This is the time since start() or lap() was last called
     * @return
     */
    public long lapTime() {
        // if started
        if(isStarted()) {
            // then add on the difference since the lap started
            
            lapTime = System.nanoTime() - startLapTimeStamp;
        }
        return lapTime;
    }
    
    /**
     * Perform a lap operation. This provides the time since the last lap call or the elapsed time so far if lap has not been called yet. (Think F1 track recording car passing start line every lap, this reports the time taken to do 1 lap of the track)
     * @return the lap time
     */
    public long lap() {
        // track the time stamp of the current lap
        long previousStartLapTimeStamp = startLapTimeStamp;
        // build a new start time stamp for the new lap
        startLapTimeStamp = System.nanoTime();
        // the difference between the time stamps is the lap time
        final long lapTime = startLapTimeStamp - previousStartLapTimeStamp;
        // add the lap time onto the total elapsed time
        elapsedTime += lapTime;
        return lapTime;
    }

    /**
     * The timeStamp of when the current lap began.
     * @return
     */
    public long lapTimeStamp() {
        if(firstLap) {
            throw new IllegalStateException("lap has not been called");
        }
        return startLapTimeStamp;
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
        super.stop();
        // force the timer to update
        lap();
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
    
    public long elapsedStopped() {
        checkStopped();
        return elapsedTime();
    }
    
    public long elapsedStarted() {
        checkStarted();
        return elapsedTime();
    }

    /**
     * reset the clock, useful post serialisation
     */
    public void resetClock() {
        startTimeStamp = System.nanoTime();
        startLapTimeStamp = startTimeStamp;
    }

    /**
     * reset time count
     */
    public void resetElapsedTime() {
        elapsedTime = 0;
    }

    /**
     * reset time count
     */
    public void resetLapTime() {
        lapTime = -1;
        startLapTimeStamp = -1;
    }

    @Override public void reset() {
        super.reset();
        resetElapsedTime();
        resetLapTime();
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
            "time=" + elapsedTime +
            ", " + super.toString() +
            ", timeStamp=" + startTimeStamp +
            '}';
    }
}
