package tsml.classifiers.distance_based.utils.system.memory;

import com.google.common.testing.GcFinalization;
import com.sun.management.GarbageCollectionNotificationInfo;

import javax.management.NotificationEmitter;
import javax.management.NotificationListener;
import javax.management.openmbean.CompositeData;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.MemoryUsage;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;
import java.util.concurrent.TimeUnit;
import tsml.classifiers.distance_based.utils.classifiers.Copy;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.timing.Stated;

/**
 * Purpose: watch the memory whilst enabled, tracking the mean, std dev, count, gc time and max mem usage.
 *
 * Note, most methods in this class are synchronized as the garbage collection updates come from another thread,
 * therefore all memory updates come from another thread and must be synced.
 *
 * Contributors: goastler
 */
public class MemoryWatcher extends Stated implements MemoryWatchable {

    public synchronized long getMaxMemoryUsageInBytes() {
        return maxMemoryUsageBytes;
    }

    private long maxMemoryUsageBytes = -1;
    private long count = 0;
    // must store the squared diff from the mean as big decimal as this number gets reallyyyyyyy big over time.
    // Downside is of course computation time, but given the updates are done on another thread this shouldn't impact
    // the main thread performance.
    private BigDecimal sqDiffFromMean = BigDecimal.ZERO;
    private double mean = 0;
    private long garbageCollectionTimeInNanos = 0;
    private transient boolean setup = false;

    public MemoryWatcher() {
        setupEmitters();
    }

    public synchronized void checkStopped() {
        super.checkStopped();
    }

    public synchronized void checkStarted() {
        super.checkStarted();
    }

    public synchronized void start(boolean check) {
        super.start(check);
    }

    public synchronized void stop(boolean check) {
        super.stop(check);
    }

    public synchronized boolean isStarted() {
        return super.isStarted();
    }

    /**
     * emitters are used to listen to each memory pool (usually young / old gen). This function sets up the emitters
     * if not already
     */
    private synchronized void setupEmitters() {
        if(!setup) {
            // garbage collector for old and young gen
            List<GarbageCollectorMXBean> garbageCollectorBeans = java.lang.management.ManagementFactory.getGarbageCollectorMXBeans();
            for (GarbageCollectorMXBean garbageCollectorBean : garbageCollectorBeans) {
                // to log
                // listen to notification from the emitter
                NotificationEmitter emitter = (NotificationEmitter) garbageCollectorBean;
                emitter.addNotificationListener(listener, null, null);
            }
            setup = true;
        }
    }

    /**
     * deserialise and setup emitters
     * @param in
     * @throws ClassNotFoundException
     * @throws IOException
     */
    private void readObject(ObjectInputStream in) throws ClassNotFoundException, IOException
    {
        try {
            Copy.setFieldValue(this, "logger", LogUtils.buildLogger(this)); // because it was transient
        } catch(NoSuchFieldException | IllegalAccessException e) {
            throw new IllegalArgumentException(e.toString()); // should never happen
        }
        in.defaultReadObject();
        setupEmitters();
    }

    private interface SerNotificationListener extends NotificationListener, Serializable {}

    /**
     * the memory update listener
     */
    private final SerNotificationListener listener = (notification, handback) -> {
        synchronized(MemoryWatcher.this) {
            if(isStarted()) {
                if (notification.getType().equals(GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION)) {
                    GarbageCollectionNotificationInfo info = GarbageCollectionNotificationInfo.from((CompositeData) notification.getUserData());
                    long duration = TimeUnit.NANOSECONDS.convert(info.getGcInfo().getDuration(), TimeUnit.MILLISECONDS);
                    garbageCollectionTimeInNanos += duration;
                    Map<String, MemoryUsage> memoryUsageInfo = info.getGcInfo().getMemoryUsageAfterGc();
                    for (Map.Entry<String, MemoryUsage> entry : memoryUsageInfo.entrySet()) {
                        MemoryUsage memoryUsageSnapshot = entry.getValue();
                        long memoryUsage = memoryUsageSnapshot.getUsed();
                        addMemoryUsageReadingInBytes(memoryUsage);
                    }
                }
            }
        }
    };

    public boolean hasReadings() {
        return count > 0;
    }

    /**
     * update the stats by adding stats from another instance
     * @param other
     */
    public synchronized void add(MemoryWatchable other) { // todo put these online std / mean algos in a util class
        maxMemoryUsageBytes = other.getMaxMemoryUsageInBytes();
        garbageCollectionTimeInNanos += other.getGarbageCollectionTimeInNanos();
        if(hasReadings() && other.hasMemoryReadings()) {
            BigDecimal thisMean = BigDecimal.valueOf(this.mean);
            BigDecimal thisCount = BigDecimal.valueOf(this.count);
            BigDecimal otherMean = BigDecimal.valueOf(other.getMeanMemoryUsageInBytes());
            BigDecimal otherCount = BigDecimal.valueOf(other.getMemoryReadingCount());
            BigDecimal overallCount = thisCount.add(otherCount);
            BigDecimal thisTotal = thisMean.multiply(thisCount);
            BigDecimal otherTotal = otherMean.multiply(otherCount);
            BigDecimal overallMean = thisTotal.add(otherTotal).divide(overallCount, RoundingMode.HALF_UP);
            // error sum of squares
            BigDecimal thisEss = BigDecimal.valueOf(getStdDevMemoryUsageInBytes()).pow(2).multiply(thisCount);
            BigDecimal otherEss = BigDecimal.valueOf(other.getStdDevMemoryUsageInBytes()).pow(2).multiply(otherCount);
            BigDecimal totalEss = thisEss.add(otherEss);
            // total group sum of squares
            BigDecimal thisTgss = thisMean.subtract(overallMean).pow(2).multiply(thisCount);
            BigDecimal otherTgss = otherMean.subtract(overallMean).pow(2).multiply(otherCount);
            BigDecimal totalTgss = thisTgss.add(otherTgss);
            // std as root of overall variance
            BigDecimal totalSqDiffFromMean = totalTgss.add(totalEss);
            mean = overallMean.doubleValue();
            count = overallCount.intValue();
            sqDiffFromMean = totalSqDiffFromMean;
        } else if(!hasReadings() && other.hasMemoryReadings()) {
            mean = other.getMeanMemoryUsageInBytes();
            count = other.getMemoryReadingCount();
            BigDecimal ess =
                BigDecimal.valueOf(other.getStdDevMemoryUsageInBytes()).pow(2).multiply(BigDecimal.valueOf(count));
            sqDiffFromMean = ess;
        } else if(hasReadings() && !other.hasMemoryReadings()) {
            // don't do anything, all our readings are already in here
        } else {
            // don't do anything here, both this and the other memory watcher have no readings
        }
    }

    /**
     * update stats from a usage reading. Beware, this is unchecked so will ignore state requirements
     * @param usage
     */
    private synchronized void addMemoryUsageReadingInBytesUnchecked(double usage) {
        maxMemoryUsageBytes = (long) Math.ceil(Math.max(maxMemoryUsageBytes, usage));
        // Welford's online algo for mean and variance
        count++;
        if(count == 1) {
            mean = usage;
        } else {
            double deltaBefore = usage - mean;
            mean += deltaBefore / count;
            double deltaAfter = usage - mean; // note the mean has changed so this isn't the same as deltaBefore
            BigDecimal bigDeltaBefore = BigDecimal.valueOf(deltaBefore);
            BigDecimal bigDeltaAfter = BigDecimal.valueOf(deltaAfter);
            BigDecimal sqDiff = bigDeltaBefore.multiply(bigDeltaAfter); // square diff from the mean
            sqDiffFromMean = sqDiffFromMean.add(sqDiff);
        }
    }

    /**
     * checked reading
     * @param usage
     */
    private synchronized void addMemoryUsageReadingInBytes(double usage) {
        if(isStarted()) {
            addMemoryUsageReadingInBytesUnchecked(usage);
        }
    }

    public synchronized double getMeanMemoryUsageInBytes() {
        if(count == 0) {
            return -1;
        }
        return mean;
    }

    public synchronized double getVarianceMemoryUsageInBytes() {
        if(count == 0) {
            return -1;
        }  else {
            return sqDiffFromMean.divide(BigDecimal.valueOf(count), BigDecimal.ROUND_HALF_UP).doubleValue(); // population variance as we see all the readings of memory usage;
        }
    }

    public synchronized long getMemoryReadingCount() {
        return count;
    }

    public double getStdDevMemoryUsageInBytes() {
        double varianceMemoryUsageInBytes = getVarianceMemoryUsageInBytes();
        if(varianceMemoryUsageInBytes < 0) {
            return varianceMemoryUsageInBytes;
        } else {
            return Math.sqrt(varianceMemoryUsageInBytes);
        }
    }

    public synchronized long getGarbageCollectionTimeInNanos() {
        return garbageCollectionTimeInNanos;
    }

    @Override
    public String toString() {
        return "MemoryWatcher{" +
            "maxMemoryUsageBytes=" + maxMemoryUsageBytes +
            ", count=" + count +
            ", mean=" + mean +
            ", garbageCollectionTimeInNanos=" + garbageCollectionTimeInNanos +
            '}';
    }

    @Override
    public synchronized void reset() {
        super.reset();
    }

    public synchronized void onReset() {
        count = 0;
        mean = 0;
        garbageCollectionTimeInNanos = 0;
        maxMemoryUsageBytes = 0;
        sqDiffFromMean = BigDecimal.ZERO;
    }

    /**
     * this cleans the memory by forcing the garbage collector invokation. Guava's finalization tools not only invoke
     * the gc but also create a weak ref and continue to invoke the gc until said weak ref is cleaned up. The theory
     * here is if the most recently created weak ref has been cleaned up, all older memory should have already been
     * dealt with / cleaned up as matter of priority over the latest memory allocation.
     */
    public void cleanup() {
        GcFinalization.awaitFullGc();
    }
}
