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

import java.util.concurrent.TimeUnit;

public interface Checkpointed extends Checkpointer {
    Checkpointer getCheckpointer();

    @Override default void setLoad(boolean state) {
        getCheckpointer().setLoad(state);
    }

    @Override default boolean isLoad() {
        return getCheckpointer().isLoad();
    }

    @Override default boolean loadCheckpoint() throws Exception {
        return getCheckpointer().loadCheckpoint();
    }

    @Override default boolean checkpointIfWorkDone() throws Exception {
        return getCheckpointer().checkpointIfWorkDone();
    }

    @Override default boolean checkpointIfIntervalExpired() throws Exception {
        return getCheckpointer().checkpointIfIntervalExpired();
    }

    @Override default boolean checkpoint() throws Exception {
        return getCheckpointer().checkpoint();
    }

    @Override default boolean isCheckpointIntervalExpired() {
        return getCheckpointer().isCheckpointIntervalExpired();
    }

    @Override default boolean isCheckpointPathSet() {
        return getCheckpointer().isCheckpointPathSet();
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

    @Override default void copyFromSerObject(Object obj) throws Exception {
        getCheckpointer().copyFromSerObject(obj);
    }
}
