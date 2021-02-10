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
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;

import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public interface Checkpointer extends Checkpointable, Loggable {

    boolean isLoad();
    void setLoad(boolean state);

    boolean loadCheckpoint() throws Exception;

    boolean checkpointIfIntervalExpired() throws Exception;

    boolean checkpoint() throws Exception;

    boolean checkpointIfWorkDone() throws Exception;

    default boolean isCheckpointIntervalExpired() {
        return getLastCheckpointTimeStamp() + getMinCheckpointIntervalNanos() < System.nanoTime();
    }

    default boolean isCheckpointPathSet() {
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

    long getLastCheckpointTimeStamp();

}
