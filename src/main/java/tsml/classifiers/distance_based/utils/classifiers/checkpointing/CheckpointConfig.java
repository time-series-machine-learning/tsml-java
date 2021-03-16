package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import tsml.classifiers.distance_based.utils.system.logging.Loggable;

import java.io.Serializable;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public class CheckpointConfig implements Serializable, Loggable {
    public CheckpointConfig() {
        setCheckpointInterval(TimeUnit.NANOSECONDS.convert(4, TimeUnit.HOURS));
        setKeepPastCheckpoints(false);
        setCheckpointPath(null);
        setLogger(null);
        clear();
    }

    private boolean keepPastCheckpoints;
    private String checkpointPath;
    private long checkpointInterval;
    private long lastCheckpointTimeStamp;
    private long checkpointLoadTime;
    private long checkpointSaveTime;
    private transient Logger logger;
    
    public void clear() {
        lastCheckpointTimeStamp = 0;
        checkpointLoadTime = 0;
        checkpointSaveTime = 0;
    }
    
    public boolean isKeepPastCheckpoints() {
        return keepPastCheckpoints;
    }

    public void setKeepPastCheckpoints(final boolean keepPastCheckpoints) {
        this.keepPastCheckpoints = keepPastCheckpoints;
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
    
    public long getLastCheckpointTimeStamp() {
        return lastCheckpointTimeStamp;
    }

    public void setLastCheckpointTimeStamp(final long lastCheckpointTimeStamp) {
        this.lastCheckpointTimeStamp = lastCheckpointTimeStamp;
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
