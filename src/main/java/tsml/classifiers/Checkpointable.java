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

import tsml.classifiers.distance_based.utils.checkpointing.CheckpointUtils;
import tsml.classifiers.distance_based.utils.classifier_mixins.Copy;

import java.io.*;
import java.util.concurrent.TimeUnit;

/**
 * Interface that allows the user to allow a classifier to checkpoint, i.e. 
save its current state and then load it again to continue building the model on 
a separate run.

By default this involves simply saving and loading a serialised the object 

known classifiers: none

Requires two methods 
number 

 * @author Tony Bagnall 2018, goastler
 */
public interface Checkpointable extends Serializable, Copy {

    /**
     * Store the path to write checkpoint files,
     * @param path
     * @return
     */
    boolean setSavePath(String path);

        /**
         Utility function to set the file structure up if required
         */
    default boolean createDirectories(String path){
        File f = new File(path);
        boolean success=true;
        if(!f.isDirectory())
            success=f.mkdirs();
        if(!isCheckpointLoadingEnabled()) {
            setLoadPath(path); // load path will be the same as the save path if not explicitly set
        }
        return success;
    }

    // save path for checkpoints. If this returns null then checkpointing is disabled
    default String getSavePath() {
        return null;
    }
    //Define how to copy from a loaded object to this object
    default void copyFromSerObject(Object obj) throws Exception {
        shallowCopyFrom(obj, CheckpointUtils.findSerFields(obj));
    }



    /**
     * set path to load checkpoints from
     * @param path
     * @return
     */
    default boolean setLoadPath(String path) {
        File f = new File(path);
        boolean success=true;
        if(!f.isDirectory())
            success=f.mkdirs();
        return success;
    }

    /**
     * get load path
     * @return
     */
    default String getLoadPath() {
        return null;
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

    /**
     * save to checkpoint function. Useful for if the classifier can be saved from an external call. This would most
     * likely happen when the classifier has finished building and we want to save the checkpoint to a "finish area"
     * to be passed onto someone / something else as a prebuilt model.
     * @return
     * @throws Exception
     */
    default boolean saveToCheckpoint() throws
                              Exception {
        throw new UnsupportedOperationException();
    }

    /**
     * same as saveToCheckpoint only we can externally ask the classifier to load from file. Both of these functions
     * provide utility methods in CheckpointUtils.
     * @return
     * @throws Exception
     */
    default boolean loadFromCheckpoint() throws Exception {
        throw new UnsupportedOperationException();
    }

    /**
     * interval between checkpointing
     * @return
     */
    default long getMinCheckpointIntervalNanos() {
        return TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    }

    /**
     * set the interval between checkpointing
     * @param minCheckpointInterval
     */
    default void setMinCheckpointIntervalNanos(final long minCheckpointInterval) {
        throw new UnsupportedOperationException();
    }

    /**
     * set the interval through timeunit rather than nanos
     * @param amount
     * @param unit
     */
    default void setMinCheckpointInterval(long amount, TimeUnit unit) {
        setMinCheckpointIntervalNanos(TimeUnit.NANOSECONDS.convert(amount, unit));
    }

    /**
     * check whether checkpoint saving is enabled. We need a check for this as just because the save path is set
     * doesn't mean the classifier should use it
     * @return
     */
    default boolean isCheckpointSavingEnabled() {
        return getSavePath() != null;
    }

    /**
     * check whether checkpoint loading is enabled. We need a check for this as just because the load path is set
     * doesn't mean the classifier should use it
     * @return
     */
    default boolean isCheckpointLoadingEnabled() {
        return getLoadPath() != null;
    }

    /**
     * simple check to see whether another checkpoint needs to be made
     * @return
     */
    default boolean hasCheckpointIntervalElapsed() {
        long diff = System.nanoTime() - getLastCheckpointTimeStamp();
        return getMinCheckpointIntervalNanos() < diff;
    }

    /**
     * get the last checkpoint timestamp, used in above func for checking checkpoint intervals
     * @return
     */
    default long getLastCheckpointTimeStamp() {
        return 0;
    }

    /**
     * call when you've done a checkpoint so the interval check will work
     * @param nanos
     */
    default void setLastCheckpointTimeStamp(final long nanos) {

    }

    /**
     * whether to skip final checkpoint. This is useful if you're going to save the final model manually and *don't*
     * want the classifier to do so itself. This often happens in ensemble classifiers where it's fine for the
     * constituent to checkpoint along the way but the ensemble should manage saving the final model.
     * @return
     */
    default boolean isSkipFinalCheckpoint() {
        return false;
    }

    /**
     * whether to skip the final checkpoint.
     * @param state
     */
    default void setSkipFinalCheckpoint(boolean state) {

    }

    long DEFAULT_MIN_CHECKPOINT_INTERVAL = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
}
