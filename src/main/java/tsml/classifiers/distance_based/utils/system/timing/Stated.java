package tsml.classifiers.distance_based.utils.system.timing;

import java.io.Serializable;

/**
 * Purpose: template for managing simple boolean start / stopped state.
 * <p>
 * Contributors: goastler
 */
public class Stated implements Serializable {

    private transient boolean started = false;

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
        final boolean wasStarted = started;
        stop(false);
        onReset();
        if(wasStarted) {
            start();
        }
    }

    public void resetAndStart() {
        reset();
        start(false);
    }

    public void resetAndStop() {
        reset();
        stop(false);
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
