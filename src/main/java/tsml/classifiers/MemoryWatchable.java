package tsml.classifiers;

import utilities.MemoryWatcher;

public interface MemoryWatchable {
    default MemoryWatcher getMemoryWatcher() { throw new UnsupportedOperationException(); }
    default long getMaxMemoryUsageInBytes() { return getMemoryWatcher().getMaxMemoryUsageInBytes(); };
    default long getMeanMemoryUsageInBytes() { return getMemoryWatcher().getMeanMemoryUsageInBytes(); };
    default long getVarianceMemoryUsageInBytes() { return getMemoryWatcher().getVarianceMemoryUsageInBytes(); };
    default long getGarbageCollectionTimeInMillis() { return getMemoryWatcher().getGarbageCollectionTimeInMillis(); };
}
