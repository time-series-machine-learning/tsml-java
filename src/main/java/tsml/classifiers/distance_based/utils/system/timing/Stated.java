package tsml.classifiers.distance_based.utils.system.timing;

import java.io.Serializable;

/**
 * Purpose: template for managing simple boolean start / stopped state.
 * <p>
 * Contributors: goastler
 */
public class Stated implements Serializable {

    private boolean started = false;

    public Stated() {}
    
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
            beforeStart();
            started = true;
            afterStart();
        } else if(check) {
            throw new IllegalStateException("already started");
        }
    }

    protected void beforeStart() {

    }

    protected void afterStart() {

    }

    public void start() {
        start(true);
    }

    public void stop(boolean check) {
        if(started) {
            beforeStop();
            started = false;
            afterStop();
        } else if(check) {
            throw new IllegalStateException("already stopped");
        }
    }

    protected void beforeStop() {

    }

    protected void afterStop() {

    }

    public void stop() {
        stop(true);
    }

    public void reset() {
        onReset();
    }

    public void resetAndStart() {
        reset();
        start(false);
    }

    public void resetAndStop() {
        stop(false);
        reset();
    }

    protected void onReset() {

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
