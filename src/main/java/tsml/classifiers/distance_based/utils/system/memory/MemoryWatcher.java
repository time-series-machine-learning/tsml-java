package tsml.classifiers.distance_based.utils.system.memory;

import com.google.common.testing.GcFinalization;
import com.sun.management.GarbageCollectionNotificationInfo;

import javax.management.ListenerNotFoundException;
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
import java.util.logging.Logger;
import tsml.classifiers.distance_based.utils.classifier_mixins.Copy;
import tsml.classifiers.distance_based.utils.stopwatch.Stated;
import tsml.classifiers.distance_based.utils.logging.LogUtils;
import tsml.classifiers.distance_based.utils.logging.Loggable;
import tsml.classifiers.distance_based.utils.stopwatch.StopWatch;

/**
 * Purpose: watch the memory whilst enabled, tracking the mean, std dev, count, gc time and max mem usage.
 *
 * Note, most methods in this class are synchronized as the garbage collection updates come from another thread,
 * therefore all memory updates come from another thread and must be synced.
 *
 * Contributors: goastler
 */
public class MemoryWatcher extends Stated implements Loggable, Serializable, MemoryWatchable {

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
    private transient Set<MemoryWatcher> listeners = new HashSet<>();

    /**
     * pass state changes to another MemoryWatcher.
     * @param other
     */
    public void addListener(MemoryWatcher other) {
        listeners.add(other);
        super.addListener(other);
    }

    public void removeListener(MemoryWatcher other) {
        listeners.remove(other);
        super.removeListener(other);
    }

    /**
     * enable if not already
     * @return
     */
    @Override public synchronized boolean enableAnyway() {
        if(super.enableAnyway() && !isEmittersSetup()) {
            setupEmitters();
            return true;
        }
        return false;
    }

    @Override public synchronized boolean enable() {
        return super.enable();
    }

    /**
     * disable if not already
     * @return
     */
    @Override public synchronized boolean disableAnyway() {
        return super.disableAnyway();
    }

    @Override public synchronized boolean disable() {
        return super.disable();
    }

    public void resetAndEnable() {
        disableAnyway();
        reset();
        enable();
    }
    // todo redundancy between this and StopWatch. We can combine the reset funcs.
    public void resetAndDisable() {
        disableAnyway();
        reset();
    }

    /**
     * emitters are used to listen to each memory pool (usually young / old gen). This function sets up the emitters
     * if not already
     */
    private synchronized void setupEmitters() {
        if(emitters == null) {
            emitters = new ArrayList<>();
            // garbage collector for old and young gen
            List<GarbageCollectorMXBean> garbageCollectorBeans = java.lang.management.ManagementFactory.getGarbageCollectorMXBeans();
            logger.finest("Setting up listeners for garbage collection ");
            for (GarbageCollectorMXBean garbageCollectorBean : garbageCollectorBeans) {
                // to log
                // listen to notification from the emitter
                NotificationEmitter emitter = (NotificationEmitter) garbageCollectorBean;
                emitters.add(emitter);
                emitter.addNotificationListener(listener, null, null);
            }
        }
    }

    /**
     * tear down emitters
     */
    private synchronized void tearDownEmitters() {
        if(emitters != null) {
            logger.finest("tearing down listeners for garbage collection");
            for(NotificationEmitter emitter : emitters) {
                try {
                    emitter.removeNotificationListener(listener);
                } catch (ListenerNotFoundException e) {
                    throw new IllegalStateException("already removed somehow...");
                }
            }
        }
    }

    /**
     * just check whether we're listening to memory updates
     * @return
     */
    private synchronized boolean isEmittersSetup() {
        return emitters != null;
    }

    // the list of emitters. They're transient as there's no point in storing emitters for a previous setup in storage.
    private transient List<NotificationEmitter> emitters;

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
            if(isEnabled()) {
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

    public MemoryWatcher() {
        super();
        reset();
        setupEmitters();
    }

    public MemoryWatcher(State state) {
        super(state);
        reset();
    }

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
        for(MemoryWatcher listener : listeners) {
            listener.add(other);
        }
    }

    /**
     * update stats from a usage reading. Beware, this is unchecked so will ignore state requirements
     * @param usage
     */
    private synchronized void addMemoryUsageReadingInBytesUnchecked(double usage) {
        logger.finest(() -> "memory reading: " + usage);
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
        if(isEnabled()) {
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

    @Override public String toString() {
        return "MemoryWatcher{" +
            super.toString() +
            ", maxMemoryUsageBytes=" + maxMemoryUsageBytes +
            '}';
    }

    public synchronized void reset() {
        disableAnyway();
        count = 0;
        mean = 0;
        garbageCollectionTimeInNanos = 0;
        maxMemoryUsageBytes = 0;
        sqDiffFromMean = BigDecimal.ZERO;
    }

    public static void main(String[] args) {
//        MemoryWatcher a = new MemoryWatcher();
//        MemoryWatcher b = new MemoryWatcher();
//        for(Double d : Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 6.0, 3.0, 3.0, 8.0, 7.0, 8.0, 6.0, 5.0, 5.0, 6.0, 5.0)) {
//            a.addMemoryUsageReadingInBytesUnchecked(d);
//        }
//        for(Double d : Arrays.asList(6.0, 3.0, 5.0, 7.0, 8.0, 9.0, 9.0, 8.0, 7.0, 8.0, 7.0, 8.0, 7.0)) {
//            b.addMemoryUsageReadingInBytesUnchecked(d);
//        }
//        System.out.println(a.getMeanMemoryUsageInBytes());
//        System.out.println(b.getMeanMemoryUsageInBytes());
//        System.out.println(a.getVarianceMemoryUsageInBytes());
//        System.out.println(b.getVarianceMemoryUsageInBytes());
//        a.add(b);
//        System.out.println(a.getMeanMemoryUsageInBytes());
//        System.out.println(a.getVarianceMemoryUsageInBytes());

        StopWatch stopWatch = new StopWatch();
        MemoryWatcher realMemWatcher = new MemoryWatcher();
        realMemWatcher.enable();
        MemoryWatcher memoryWatcher = new MemoryWatcher();
        stopWatch.enable();
        Random rand = new Random(0);
        int max = 1_000_000;
        for(int i = 0; i < max; i++) {
            memoryWatcher.addMemoryUsageReadingInBytesUnchecked(Math.abs(rand.nextInt(100)));
            if(rand.nextInt(max) < 10_000) {
                System.out.println("gc");
                System.gc();
            }
        }
        stopWatch.disable();
        realMemWatcher.disable();
        System.out.println(realMemWatcher.getMaxMemoryUsageInBytes());
        System.out.println(realMemWatcher.getMeanMemoryUsageInBytes());
        System.out.println(realMemWatcher.getStdDevMemoryUsageInBytes());
        System.out.println(realMemWatcher.getGarbageCollectionTimeInNanos());
        System.out.println("----");
        System.out.println(stopWatch.getTimeNanos());
        System.out.println(TimeUnit.SECONDS.convert(stopWatch.getTimeNanos(), TimeUnit.NANOSECONDS));
    }

    private transient final Logger logger = LogUtils.buildLogger(this);

    @Override public Logger getLogger() {
        return logger;
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
