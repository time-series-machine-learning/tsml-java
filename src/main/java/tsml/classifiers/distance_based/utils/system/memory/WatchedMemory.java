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
