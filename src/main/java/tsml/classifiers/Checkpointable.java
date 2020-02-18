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

import utilities.CheckpointUtils;
import utilities.Copy;
import utilities.FileUtils;

import java.io.*;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.function.Predicate;

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
        if(!isCheckpointLoadingEnabled()) {
            setLoadPath(path);
        }
        return success;
    }

    // save path for checkpoints. If this returns null then checkpointing is disabled
    default String getSavePath() {
        return null;
    }
    //Define how to copy from a loaded object to this object
    default void copyFromSerObject(Object obj) throws Exception {
        shallowCopyFrom(obj, findSerFields(obj));
    }

    default boolean setLoadPath(String path) {
        File f = new File(path);
        boolean success=true;
        if(!f.isDirectory())
            success=f.mkdirs();
        return success;
    }

    default String getLoadPath() {
        return null;
    }

    static Set<Field> findSerFields(Object obj) {
        return Copy.findFields(obj.getClass(), TRANSIENT.negate().and(Copy.DEFAULT_FIELDS));
    }

    //Override both if not using Java serialisation    
    default void saveToFile(String filename) throws Exception {
        CheckpointUtils.serialise(this, filename);
    }
    default void loadFromFile(String filename) throws Exception{
        Object obj = CheckpointUtils.deserialise(filename);
        if(obj != null) {
            copyFromSerObject(obj);
        }
    }

    default boolean saveToCheckpoint() throws
                              Exception {
        throw new UnsupportedOperationException();
    }

    default boolean loadFromCheckpoint() throws Exception {
        throw new UnsupportedOperationException();
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

    default boolean isCheckpointSavingEnabled() {
        return getSavePath() != null;
    }

    default boolean isCheckpointLoadingEnabled() {
        return getLoadPath() != null;
    }

    default boolean hasCheckpointIntervalElapsed() {
        long diff = System.nanoTime() - getLastCheckpointTimeStamp();
        return getMinCheckpointIntervalNanos() < diff;
    }

    default long getLastCheckpointTimeStamp() {
        return 0;
    }

    default void setLastCheckpointTimeStamp(final long nanos) {

    }

    default boolean isSkipFinalCheckpoint() {
        return false;
    }

    default void setSkipFinalCheckpoint(boolean state) {

    }

    Predicate<Field> TRANSIENT = field -> Modifier.isTransient(field.getModifiers());
    long DEFAULT_MIN_CHECKPOINT_INTERVAL = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
}
