package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import tsml.classifiers.Checkpointable;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import utilities.FileUtils;

import java.io.File;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public interface Chkpt extends Checkpointable {

    String CHECKPOINT_EXTENSION = "tar.gz";
    String CHECKPOINT_EXTENSION_WITH_DOT = "." + CHECKPOINT_EXTENSION;
    String CHECKPOINT_PREFIX = "checkpoint";
    String CHECKPOINT_PREFIX_WITH_UNDERSCORE = CHECKPOINT_PREFIX + "_";
    
    CheckpointConfig getCheckpointConfig();
    
    default String getCheckpointPath() {
        return getCheckpointConfig().getCheckpointPath();
    }
    
    default long getLastCheckpointTimeStamp() {
        return getCheckpointConfig().getLastCheckpointTimeStamp();
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
    
    default boolean isKeepPastCheckpoints() {
        return getCheckpointConfig().isKeepPastCheckpoints();
    }
    
    default void setKeepPastCheckpoints(boolean state) {
        getCheckpointConfig().setKeepPastCheckpoints(state);
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
        config.setLastCheckpointTimeStamp(System.nanoTime());
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
                final String checkpointPath = path + "/" + CHECKPOINT_PREFIX_WITH_UNDERSCORE + System.nanoTime() + CHECKPOINT_EXTENSION_WITH_DOT;
                final long timeStampBeforeSave = System.nanoTime();
                // take snapshot of timings
                getCheckpointConfig().addSaveTime(timeStampBeforeSave - timeStamp);
                // update the start time as we've already accounted for time before save operation
                timeStamp = timeStampBeforeSave;
                saveToFile(checkpointPath);
                // remove any previous checkpoints
                if(!isKeepPastCheckpoints()) {
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
                getCheckpointConfig().setLastCheckpointTimeStamp(System.nanoTime());
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
        return System.nanoTime() >= getLastCheckpointTimeStamp() + getCheckpointInterval();
    }
}
