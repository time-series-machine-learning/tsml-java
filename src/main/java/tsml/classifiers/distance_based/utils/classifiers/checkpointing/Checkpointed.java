package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import tsml.classifiers.Checkpointable;
import tsml.classifiers.distance_based.utils.classifiers.Copier;
import utilities.FileUtils;

import java.util.concurrent.TimeUnit;

public interface Checkpointed extends Checkpointable, Copier {

    String DEFAULT_CHECKPOINT_FILENAME = "checkpoint.gz.ser";
    long DEFAULT_CHECKPOINT_INTERVAL = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);

    /**
     * Get the time spent checkpointing. Classifiers which grow to occupy large amount of mem take a lot of time to save to disk. Therefore it is helpful to record this for consideration in timing conclusions.
     * @return <0 for time not recorded, >=0 for the time taken checkpointing so far.
     */
    default long getCheckpointTime() {
        return -1;
    }
    
    long getCheckpointInterval();
    void setCheckpointInterval(long nanos);
    default void setCheckpointInterval(long amount, TimeUnit unit) {
        setCheckpointInterval(TimeUnit.NANOSECONDS.convert(amount, unit));
    }
    @Override default boolean setCheckpointTimeHours(int t) {
        setCheckpointInterval(t, TimeUnit.HOURS);
        return true;
    }
    
    default boolean isCheckpointIntervalExpired() {
        return getLastCheckpointTimeStamp() + getCheckpointInterval() < System.nanoTime();
    }

    boolean isCheckpointLoadingEnabled();
    void setCheckpointLoadingEnabled(boolean state);
    
    long getLastCheckpointTimeStamp();
    void setLastCheckpointTimeStamp(long timeStamp);
    
    String getCheckpointPath();
    
    String getCheckpointFileName();
    void setCheckpointFileName(String name);
    
    default boolean loadCheckpoint() {
        if(!isCheckpointLoadingEnabled() || !isCheckpointPathValid()) {
            return false;
        }
        try {
            String path = getCheckpointPath() + "/" + getCheckpointFileName();
            loadFromFile(path);
            setLastCheckpointTimeStamp(System.nanoTime());
            // disable loading future checkpoints (i.e. this has loaded a checkpoint already, any further checkpoints should be produced by this instance therefore pointless reloading the progress achieved by this instance)
            setCheckpointLoadingEnabled(false);
            return true;
        } catch(Exception e) {
            return false;
        }
    }
    
    default boolean forceSaveCheckpoint() {
        return saveCheckpoint(true);
    }
    
    default boolean saveCheckpoint() {
        return saveCheckpoint(false);
    }
    
    default boolean saveCheckpoint(boolean force) {
        if(!isCheckpointPathValid()) {
            return false;
        }
        if(!force && isCheckpointIntervalExpired()) {
            return false;
        }
        try {
            String path = getCheckpointPath() + "/" + getCheckpointFileName();
            saveToFile(path);
            setLastCheckpointTimeStamp(System.nanoTime());
            return true;
        } catch(Exception e) {
            return false;
        }
    }
    
    default boolean isCheckpointPathValid() {
        return getCheckpointPath() != null;
    }
    
    @Override default void loadFromFile(String path) throws Exception {
        Checkpointable.super.loadFromFile(path);
    }

    @Override default void saveToFile(String path) throws Exception {
        FileUtils.makeParentDir(path);
        Checkpointable.super.saveToFile(path);
    }

    @Override default void copyFromSerObject(Object obj) throws Exception {
        shallowCopyFrom(obj);
    }
}
