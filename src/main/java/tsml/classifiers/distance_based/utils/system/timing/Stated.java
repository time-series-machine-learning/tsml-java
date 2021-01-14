/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
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
