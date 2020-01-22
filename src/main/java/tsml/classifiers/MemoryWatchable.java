package tsml.classifiers;

import utilities.MemoryWatcher;

public interface MemoryWatchable {
    default MemoryWatcher getMemoryWatcher() {
        throw new UnsupportedOperationException();
    }
    default long getMaxMemoryUsageInBytes() { return getMemoryWatcher().getMaxMemoryUsageInBytes(); };
    default double getMeanMemoryUsageInBytes() { return getMemoryWatcher().getMeanMemoryUsageInBytes(); };
    default double getVarianceMemoryUsageInBytes() { return getMemoryWatcher().getVarianceMemoryUsageInBytes(); };
    default double getStdDevMemoryUsageInBytes() { return getMemoryWatcher().getStdDevMemoryUsageInBytes(); }
    default long getGarbageCollectionTimeInMillis() { return getMemoryWatcher().getGarbageCollectionTimeInMillis(); };
}
