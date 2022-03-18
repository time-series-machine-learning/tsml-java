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
 
package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import tsml.classifiers.Checkpointable;
import tsml.classifiers.distance_based.utils.classifiers.contracting.TimedTrain;
import tsml.classifiers.distance_based.utils.experiment.TimeSpan;
import tsml.classifiers.distance_based.utils.system.copy.CopierUtils;
import utilities.FileUtils;

import java.io.File;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public interface Checkpointed extends Checkpointable, TimedTrain {

    String CHECKPOINT_EXTENSION = "tar.gz";
    String CHECKPOINT_EXTENSION_WITH_DOT = "." + CHECKPOINT_EXTENSION;
    String CHECKPOINT_PREFIX = "checkpoint";
    String CHECKPOINT_PREFIX_WITH_UNDERSCORE = CHECKPOINT_PREFIX + "_";
    
    CheckpointConfig getCheckpointConfig();
    
    default String getCheckpointPath() {
        return getCheckpointConfig().getCheckpointPath();
    }

    /**
     * Get the run time at the last checkpoint. We use this to work out whether the checkpoint interval has expired /
     * allocate checkpoints to intervals in the case where checkpoints do not land exactly in the interval time points.
     * @return
     */
    default long getLastCheckpointRunTime() {
        return getCheckpointConfig().getLastCheckpointRunTime();
    }

    /**
     * time spent saving checkpoints
     */
    default long getCheckpointSaveTime() {
        return getCheckpointConfig().getCheckpointSaveTime();
    }
    
    /**
     * time spent saving checkpoints
     */
    default long getCheckpointLoadTime() {
        return getCheckpointConfig().getCheckpointSaveTime();
    }
    
    default long getCheckpointInterval() {
        return getCheckpointConfig().getCheckpointInterval();
    }
    
    default void setCheckpointInterval(final long checkpointInterval) {
        getCheckpointConfig().setCheckpointInterval(checkpointInterval);
    }

    default void setCheckpointInterval(final long amount, final TimeUnit unit) {
        setCheckpointInterval(TimeUnit.NANOSECONDS.convert(amount, unit));
    }
    
    default void setCheckpointInterval(final TimeSpan timeSpan) {
        setCheckpointInterval(timeSpan.inNanos());
    }

    default boolean isCheckpointPathSet() {
        return getCheckpointConfig().getCheckpointPath() != null;
    }


    /**
     * the total time spent checkpointing
     * @return
     */
    default long getCheckpointingTime() {
        return getCheckpointLoadTime() + getCheckpointSaveTime();
    }
    
    default boolean isKeepCheckpoints() {
        return getCheckpointConfig().isKeepCheckpoints();
    }
    
    default void setKeepCheckpoints(boolean state) {
        getCheckpointConfig().setKeepCheckpoints(state);
    }

    /**
     * Load the most recent checkpoint
     * @return
     */
    default boolean loadCheckpoint() throws Exception {
        final long startTimeStamp = System.nanoTime();
        boolean loaded = false;
        if(isCheckpointPathSet()) {
            final Logger logger = getCheckpointConfig().getLogger();
            final File dir = new File(getCheckpointPath());
            if(!dir.exists()) {
                logger.info("checkpoint dir does not exist, skipping load checkpoint");
            } else if(!dir.isDirectory()) {
                logger.info("checkpoint path is not a dir, skipping load checkpoint");
            } else {
                final String[] files = dir.list();
                if(files == null || files.length <= 0) {
                    logger.info("no past checkpoints found");
                } else {
                    // get the file with the largest timestamp. Files are saved with as <timestamp>.tar.gz
                    Arrays.sort(files, Comparator.comparingLong(file -> {
                        final String time = file.replace(CHECKPOINT_PREFIX_WITH_UNDERSCORE, "").replace(CHECKPOINT_EXTENSION_WITH_DOT, "");
                        return Long.parseLong(time);
                    }));
                    final String file = files[files.length - 1];
                    final String checkpointPath = dir + "/" + file;
                    loadFromFile(checkpointPath);
                    loaded = true;
                    logger.info("loaded checkpoint from " + checkpointPath);
                }
            }
        }

        // add time taken loading checkpoints onto total load time
        CheckpointConfig config = getCheckpointConfig();
        // set the last checkpoint time to now to avoid saveCheckpoint calls after this saving a new checkpoint when
        // less than interval time has passed
        config.setLastCheckpointRunTime(getRunTime());
        config.addLoadTime(System.nanoTime() - startTimeStamp);
        return loaded;
    }
    
    default boolean saveCheckpoint() throws Exception {
        return saveCheckpoint(false);
    }

    /**
     * Save checkpoint irrelevant of checkpoint interval
     * @return
     * @throws Exception
     */
    default boolean forceSaveCheckpoint() throws Exception {
        return saveCheckpoint(true);
    }
    
    default boolean saveCheckpoint(boolean force) throws Exception {
        long timeStamp = System.nanoTime();
        boolean saved = false;
        if(isCheckpointPathSet()) {
            if(isCheckpointIntervalExpired() || force) {
                // get current checkpoints that already exist
                final String path = getCheckpointPath();
                final String[] files = new File(path).list();
                // save this checkpoint
                final long runTime = getRunTime();
                final String checkpointPath = path + "/" + CHECKPOINT_PREFIX_WITH_UNDERSCORE + runTime + CHECKPOINT_EXTENSION_WITH_DOT;
                final long timeStampBeforeSave = System.nanoTime();
                // take snapshot of timings
                getCheckpointConfig().addSaveTime(timeStampBeforeSave - timeStamp);
                // update the start time as we've already accounted for time before save operation
                timeStamp = timeStampBeforeSave;
                saveToFile(checkpointPath);
                // remove any previous checkpoints
                if(!isKeepCheckpoints()) {
                    if(files != null) {
                        for(String file : files) {
                            final File f = new File(path + "/" + file);
                            final String name = f.getName();
                            if(name.startsWith(CHECKPOINT_PREFIX_WITH_UNDERSCORE) && name.endsWith(CHECKPOINT_EXTENSION_WITH_DOT)) {
                                if(!f.delete()) {
                                    throw new IllegalStateException("failed to delete checkpoint " + f.getPath());
                                }
                            }
                        }
                    }
                }
                getCheckpointConfig().getLogger().info("saved checkpoint to " + checkpointPath);
                // update the checkpoint time stamp
                getCheckpointConfig().setLastCheckpointRunTime(runTime);
            }
        }
        getCheckpointConfig().addSaveTime(System.nanoTime() - timeStamp);
        return saved;
    }
    
    @Override default void copyFromSerObject(Object obj) throws Exception {
        CopierUtils.shallowCopy(obj, this);
    }

    @Override default void saveToFile(String path) throws Exception {
        FileUtils.makeParentDir(path);
        Checkpointable.super.saveToFile(path);
    }

    @Override default boolean setCheckpointPath(String path) {
        getCheckpointConfig().setCheckpointPath(path);
        return true;
    }


    default boolean isCheckpointIntervalExpired() {
        // need to work out what interval we're in. E.g. say the last checkpoint was at 11hrs and we've got an interval
        // of 3hrs. The checkpoint should have occurred at 9hrs, but was missed due to processing / when checkpoints can
        // be taken. Therefore, the checkpoint occurred at 11hrs. We still want to checkpoint at 12hrs, otherwise the
        // pattern of intervals is ever changing. I.e. we want 3hrs, 6hrs, 9hrs, 12hrs, ... or as close to that as
        // possible. Therefore, we'll work out what interval the last checkpoint time corresponds to (e.g. 11hr
        // corresponding to the 9hr interval point) and find the next interval point from there (e.g. 9hr + 3hr = 12hr)
        final long lastCheckpointTimeStamp = getLastCheckpointRunTime();
        final long checkpointInterval = getCheckpointInterval();
        final long startTimeStampForMostRecentInterval =  lastCheckpointTimeStamp - lastCheckpointTimeStamp % checkpointInterval;
        return getRunTime() >= startTimeStampForMostRecentInterval + checkpointInterval;
    }
}
