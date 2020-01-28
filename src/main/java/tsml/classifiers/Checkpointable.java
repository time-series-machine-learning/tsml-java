/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.classifiers;

import utilities.Copy;
import utilities.FileUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.concurrent.TimeUnit;

/**
 * Interface that allows the user to allow a classifier to checkpoint, i.e. 
save its current state and then load it again to continue building the model on 
a separate run.

By default this involves simply saving and loading a serialised the object 

known classifiers: none

Requires two methods 
number 

 * @author Tony Bagnall 2018
 */
public interface Checkpointable extends Serializable, Copy {

    //Set the path where checkpointed versions will be stored
    default boolean setSavePath(String path){
        File f = new File(path);
        boolean success=true;
        if(!f.isDirectory())
            success=f.mkdirs();
        return success;
    }

    // save path for checkpoints. If this returns null then checkpointing is disabled
    default String getSavePath() {
        return null;
    }
    //Define how to copy from a loaded object to this object
    default void copyFromSerObject(Object obj) throws Exception {
        shallowCopyFrom(obj);
    }

    //Override both if not using Java serialisation    
    default void saveToFile(String filename) throws IOException{
        FileUtils.makeParentDir(filename);
        FileOutputStream fos =
        new FileOutputStream(filename);
        try (ObjectOutputStream out = new ObjectOutputStream(fos)) {
            out.writeObject(this);
            out.close();
            fos.close();
        }
    }
    default void loadFromFile(String filename) throws Exception{
        FileInputStream fis = new FileInputStream(filename);
        try (ObjectInputStream in = new ObjectInputStream(fis)) {
            Object obj=in.readObject();
            copyFromSerObject(obj);
        }
    }

    default void checkpoint() throws
                              Exception {
        throw new UnsupportedOperationException();
    }

    default boolean isCheckpointing() {
        return getSavePath() != null;
    }

    default long getMinCheckpointIntervalNanos() {
        return TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    }

    default void setMinCheckpointIntervalNanos(final long minCheckpointInterval) {
        throw new UnsupportedOperationException();
    }

    default void setMinCheckpointInterval(long amount, TimeUnit unit) {
        setMinCheckpointIntervalNanos(TimeUnit.NANOSECONDS.convert(amount, unit));
    }

    default boolean isIgnorePreviousCheckpoints() {
        return false;
    }

    default void setIgnorePreviousCheckpoints(boolean state) {
        throw new UnsupportedOperationException();
    }
    
}
