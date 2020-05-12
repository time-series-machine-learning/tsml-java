package tsml.classifiers.distance_based.proximity;

import tsml.classifiers.MemoryContractable;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class MemoryContracter implements MemoryContractable {

    private final MemoryWatcher memoryWatcher = new MemoryWatcher();

    public MemoryContracter(final MemoryWatcher watcher) {
        this.watcher = watcher;
    }

    public MemoryContracter() {
        this(new MemoryWatcher());
    }

//    public long getRemainingMemoryInBytes() {
//        if(!hasTrainTimeLimit()) {
//            throw new IllegalStateException("time limit not set");
//        }
//        long timeTaken = watcher.getM;
//        long limit = getTrainTimeLimit();
//        long diff = limit - timeTaken;
//        return Math.max(0, limit);
//    }

//    public boolean hasRemainingTrainTime() {
//        return getRemainingTrainTime() > 0;
//    }
//
//    public boolean hasTrainTimeLimit() {
//        return getTrainTimeLimit() > 0;
//    }

    private long memoryLimitBytes = 0;
    private final MemoryWatcher watcher;

    public MemoryWatcher getWatcher() {
        return watcher;
    }

    @Override
    public void setMemoryLimit(final DataUnit unit, final long amount) {

    }
}
