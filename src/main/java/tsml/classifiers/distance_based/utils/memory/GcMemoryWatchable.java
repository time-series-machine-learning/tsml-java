package tsml.classifiers.distance_based.utils.memory;

import tsml.classifiers.distance_based.utils.MemoryWatchable;

/**
 * Purpose: Simple interface to defer memory funcs to the memory watcher. The memory watcher is the concrete
 * implementation for the memory funcs, so implementors of this interface need only define a getter for their memory
 * watcher rather than handle all of the memory funcs which are rather big and difficult to deal with. The memory
 * watcher class does the heavy lifting of tracking stats whilst this interface just wraps around it.
 *
 * Contributors: goastler
 */
public interface GcMemoryWatchable extends MemoryWatchable {
    MemoryWatcher getMemoryWatcher();
    default long getMaxMemoryUsageInBytes() { return getMemoryWatcher().getMaxMemoryUsageInBytes(); };
    default double getMeanMemoryUsageInBytes() { return getMemoryWatcher().getMeanMemoryUsageInBytes(); };
    default double getVarianceMemoryUsageInBytes() { return getMemoryWatcher().getVarianceMemoryUsageInBytes(); };
    default double getStdDevMemoryUsageInBytes() { return getMemoryWatcher().getStdDevMemoryUsageInBytes(); }
    default long getGarbageCollectionTimeInMillis() { return getMemoryWatcher().getGarbageCollectionTimeInMillis(); };
    default long getMemoryReadingCount() { return getMemoryWatcher().getMemoryReadingCount(); }
}
