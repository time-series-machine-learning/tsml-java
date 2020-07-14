package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import tsml.classifiers.Checkpointable;

import java.util.concurrent.TimeUnit;

public interface Checkpointer extends Checkpointable {

    boolean isLoadCheckpoint();
    void setLoadCheckpoint(boolean state);

    boolean loadCheckpoint() throws Exception;

    boolean saveCheckpoint(boolean force) throws Exception;

    boolean saveCheckpoint() throws Exception;

    boolean saveFinalCheckpoint() throws Exception;

    default boolean isCheckpointIntervalExpired() {
        return getLastCheckpointTimeStamp() + getMinCheckpointIntervalNanos() < System.nanoTime();
    }

    default boolean isCheckpointing() {
        return getCheckpointDirPath() != null;
    }

    String getCheckpointFileName();
    void setCheckpointFileName(String checkpointFileName);

    String getCheckpointDirPath();
    void setCheckpointDirPath(String checkpointDirPath);

    default String getCheckpointFilePath() {
        return getCheckpointDirPath() + "/" + getCheckpointFileName();
    }

    @Override default boolean setCheckpointPath(String path) {
        setCheckpointDirPath(path);
        return true;
    }

    long getMinCheckpointIntervalNanos();
    void setMinCheckpointIntervalNanos(long minCheckpointIntervalNanos);
    default void setMinCheckpointIntervalNanos(long time, TimeUnit unit) {
        setMinCheckpointIntervalNanos(TimeUnit.NANOSECONDS.convert(time, unit));
    }

    @Override boolean setCheckpointTimeHours(int t);

    @Override long getLastCheckpointTimeStamp();

}
