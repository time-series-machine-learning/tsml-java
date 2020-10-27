package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import tsml.classifiers.Checkpointable;
import tsml.classifiers.distance_based.utils.classifiers.Copier;

import java.util.concurrent.TimeUnit;

public interface Checkpointed extends Checkpointable, Copier {

    String DEFAULT_CHECKPOINT_FILENAME = "checkpoint.gz.ser";
    long DEFAULT_CHECKPOINT_INTERVAL = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    
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
    
    default boolean saveCheckpoint() {
        return saveCheckpoint(false);
    }
    
    default boolean saveCheckpoint(boolean checkInterval) {
        if(!isCheckpointPathValid()) {
            return false;
        }
        if(checkInterval && isCheckpointIntervalExpired()) {
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
    
    @Override default void loadFromFile(String filename) throws Exception {
        Checkpointable.super.loadFromFile(filename);
    }

    @Override default void saveToFile(String filename) throws Exception {
        Checkpointable.super.saveToFile(filename);
    }

    @Override default void copyFromSerObject(Object obj) throws Exception {
        shallowCopyFrom(obj);
    }
}
