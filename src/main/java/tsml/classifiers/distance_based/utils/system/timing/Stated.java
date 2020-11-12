package tsml.classifiers.distance_based.utils.system.timing;

import java.io.Serializable;

/**
 * Purpose: template for managing simple boolean start / stopped state.
 * <p>
 * Contributors: goastler
 */
public class Stated implements Serializable {

    private boolean started;

    public Stated() {
        this(false);
    }
    
    public Stated(boolean start) {
        reset();
        if(start) {
            start();
        }
    }
    
    public boolean isStarted() {
        return started;
    }

    public boolean isStopped() {
        return !started;
    }

    public void start(boolean check) {
        if(!started) {
            started = true;
        } else if(check) {
            throw new IllegalStateException("already started");
        }
    }

    public void start() {
        start(true);
    }

    public void stop(boolean check) {
        if(started) {
            started = false;
        } else if(check) {
            throw new IllegalStateException("already stopped");
        }
    }

    public void stop() {
        stop(true);
    }

    public void reset() {
        
    }

    public void resetAndStart() {
        reset();
        start(false);
        reset();
    }

    public void resetAndStop() {
        reset();
        stop(false);
        reset();
    }

    public void checkStopped() {
        if(started) {
            throw new IllegalStateException("not stopped");
        }
    }

    public void checkStarted() {
        if(!started) {
            throw new IllegalStateException("not started");
        }
    }

    @Override
    public String toString() {
        return "started=" + started;
    }
}
