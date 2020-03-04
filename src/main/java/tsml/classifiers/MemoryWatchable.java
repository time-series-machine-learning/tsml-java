package tsml.classifiers;

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
