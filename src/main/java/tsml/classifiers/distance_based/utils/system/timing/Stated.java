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

    public void start() {
        if(!started) {
            started = true;
        } else {
            throw new IllegalStateException("already started");
        }
    }
    
    public void optionalStart() {
        if(!isStarted()) {
            start();
        }
    }
    
    public void optionalStop() {
        if(!isStopped()) {
            stop();
        }
    }
    
    public void stop() {
        if(started) {
            started = false;
        } else {
            throw new IllegalStateException("already stopped");
        }
    }

    public void reset() {
        
    }

    public void resetAndStart() {
        reset();
        optionalStart();
    }

    public void stopAndReset() {
        optionalStop();
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
