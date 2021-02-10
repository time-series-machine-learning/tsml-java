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
 
package tsml.classifiers.distance_based.utils.system.memory;

/**
 * Purpose: Simple interface to defer memory funcs to the memory watcher. The memory watcher is the concrete
 * implementation for the memory funcs, so implementors of this interface need only define a getter for their memory
 * watcher rather than handle all of the memory funcs which are rather big and difficult to deal with. The memory
 * watcher class does the heavy lifting of tracking stats whilst this interface just wraps around it.
 *
 * Contributors: goastler
 */
public interface WatchedMemory extends MemoryWatchable {
    MemoryWatcher getMemoryWatcher();
    default long getMaxMemoryUsage() { return getMemoryWatcher().getMaxMemoryUsage(); };
    default double getMeanMemoryUsage() { return getMemoryWatcher().getMeanMemoryUsage(); };
    default double getVarianceMemoryUsage() { return getMemoryWatcher().getVarianceMemoryUsage(); };
    default double getStdDevMemoryUsage() { return getMemoryWatcher().getStdDevMemoryUsage(); }
    default long getGarbageCollectionTime() { return getMemoryWatcher().getGarbageCollectionTime(); };
    default long getMemoryReadingCount() { return getMemoryWatcher().getMemoryReadingCount(); }
}
