package tsml.classifiers;

public interface MemoryWatchable {
    default long getMaxMemoryUsageInBytes() { return -1; };
    default long getMeanMemoryUsageInBytes() { return -1; };
    default long getVarianceMemoryUsageInBytes() { return -1; };
    default long getGarbageCollectionTimeInMillis() { return -1; };
}
