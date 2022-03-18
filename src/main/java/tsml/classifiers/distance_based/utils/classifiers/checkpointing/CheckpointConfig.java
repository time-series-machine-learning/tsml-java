package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import tsml.classifiers.distance_based.utils.system.logging.Loggable;

import java.io.Serializable;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public class CheckpointConfig implements Serializable, Loggable {
    public CheckpointConfig() {
        setCheckpointInterval(TimeUnit.NANOSECONDS.convert(4, TimeUnit.HOURS));
        setKeepCheckpoints(false);
        setCheckpointPath(null);
        setLogger(null);
        clear();
    }

    private boolean keepCheckpoints;
    private String checkpointPath;
    private long checkpointInterval;
    private long lastCheckpointRunTime;
    private long checkpointLoadTime;
    private long checkpointSaveTime;
    private transient Logger logger;
    
    public void clear() {
        lastCheckpointRunTime = 0;
        checkpointLoadTime = 0;
        checkpointSaveTime = 0;
    }
    
    public boolean isKeepCheckpoints() {
        return keepCheckpoints;
    }

    public void setKeepCheckpoints(final boolean keepCheckpoints) {
        this.keepCheckpoints = keepCheckpoints;
    }

    public String getCheckpointPath() {
        return checkpointPath;
    }

    public boolean setCheckpointPath(final String checkpointPath) {
        this.checkpointPath = checkpointPath;
        return checkpointPath != null;
    }

    public long getCheckpointInterval() {
        return checkpointInterval;
    }

    public void setCheckpointInterval(final long checkpointInterval) {
        this.checkpointInterval = checkpointInterval;
    }
    
    public long getLastCheckpointRunTime() {
        return lastCheckpointRunTime;
    }

    public void setLastCheckpointRunTime(final long lastCheckpointRunTime) {
        this.lastCheckpointRunTime = lastCheckpointRunTime;
    }

    public long getCheckpointLoadTime() {
        return checkpointLoadTime;
    }
    
    public long getCheckpointSaveTime() {
        return checkpointSaveTime;
    }
    
    public void addSaveTime(long time) {
        checkpointSaveTime += time;
    }
    
    public void addLoadTime(long time) {
        checkpointLoadTime += time;
    }

    @Override public Logger getLogger() {
        return logger;
    }

    @Override public void setLogger(final Logger logger) {
        this.logger = logger;
    }
    
    public void resetSaveTime() {
        checkpointSaveTime = 0;
    }
    
    public void resetLoadTime() {
        checkpointLoadTime = 0;
    }
    
    public void resetCheckpointingTime() {
        resetLoadTime();
        resetSaveTime();
    }


}
