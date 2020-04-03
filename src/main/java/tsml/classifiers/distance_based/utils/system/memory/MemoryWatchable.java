package tsml.classifiers.distance_based.utils.system.memory;

/**
 * Purpose: get stats related to memory.
 *
 * Contributors: goastler
 */
public interface MemoryWatchable {
    long getMaxMemoryUsageInBytes();
    double getMeanMemoryUsageInBytes();
    double getVarianceMemoryUsageInBytes();
    double getStdDevMemoryUsageInBytes();
    long getGarbageCollectionTimeInMillis();
    long getMemoryReadingCount();
    default boolean hasMemoryReadings() {
        return getMemoryReadingCount() > 0;
    }
}
