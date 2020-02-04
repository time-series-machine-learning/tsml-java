package tsml.classifiers;

import utilities.MemoryWatcher;

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
