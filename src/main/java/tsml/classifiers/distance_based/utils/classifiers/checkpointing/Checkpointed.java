package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import java.util.concurrent.TimeUnit;

public interface Checkpointed extends Checkpointer {
    Checkpointer getCheckpointer();

    @Override default void setLoadCheckpoint(boolean state) {
        getCheckpointer().setLoadCheckpoint(state);
    }

    @Override default boolean isLoadCheckpoint() {
        return getCheckpointer().isLoadCheckpoint();
    }

    @Override default boolean loadCheckpoint() throws Exception {
        return getCheckpointer().loadCheckpoint();
    }

    @Override default boolean saveCheckpoint(boolean force) throws Exception {
        return getCheckpointer().saveCheckpoint(force);
    }

    @Override default boolean saveCheckpoint() throws Exception {
        return getCheckpointer().saveCheckpoint();
    }

    @Override default boolean saveFinalCheckpoint() throws Exception {
        return getCheckpointer().saveFinalCheckpoint();
    }

    @Override default boolean isCheckpointIntervalExpired() {
        return getCheckpointer().isCheckpointIntervalExpired();
    }

    @Override default boolean isCheckpointing() {
        return getCheckpointer().isCheckpointing();
    }

    @Override default String getCheckpointFileName() {
        return getCheckpointer().getCheckpointFileName();
    }

    @Override default void setCheckpointFileName(String checkpointFileName) {
        getCheckpointer().setCheckpointFileName(checkpointFileName);
    }

    @Override default String getCheckpointDirPath() {
        return getCheckpointer().getCheckpointDirPath();
    }

    @Override default void setCheckpointDirPath(String checkpointDirPath) {
        getCheckpointer().setCheckpointDirPath(checkpointDirPath);
    }

    @Override default boolean setCheckpointPath(String path) {
        return getCheckpointer().setCheckpointPath(path);
    }

    @Override default long getMinCheckpointIntervalNanos() {
        return getCheckpointer().getMinCheckpointIntervalNanos();
    }

    @Override default void setMinCheckpointIntervalNanos(long minCheckpointIntervalNanos) {
        getCheckpointer().setMinCheckpointIntervalNanos(minCheckpointIntervalNanos);
    }

    @Override default void setMinCheckpointIntervalNanos(long time, TimeUnit unit) {
        getCheckpointer().setMinCheckpointIntervalNanos(time, unit);
    }

    @Override default boolean setCheckpointTimeHours(int t) {
        return getCheckpointer().setCheckpointTimeHours(t);
    }

    @Override default long getLastCheckpointTimeStamp() {
        return getCheckpointer().getLastCheckpointTimeStamp();
    }

}
