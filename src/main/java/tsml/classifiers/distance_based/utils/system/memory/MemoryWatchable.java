package tsml.classifiers.distance_based.utils.system.memory;

/**
 * Purpose: get stats related to memory.
 *
 * Contributors: goastler
 */
public interface MemoryWatchable {
    long getMaxMemoryUsage();
    double getMeanMemoryUsage();
    double getVarianceMemoryUsage();
    double getStdDevMemoryUsage();
    long getGarbageCollectionTime();
    long getMemoryReadingCount();
    default boolean hasMemoryReadings() {
        return getMemoryReadingCount() > 0;
    }
}
