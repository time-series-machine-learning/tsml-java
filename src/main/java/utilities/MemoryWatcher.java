package utilities;

import com.sun.management.GarbageCollectionNotificationInfo;
import tsml.classifiers.MemoryWatchable;

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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class MemoryWatcher extends Stated implements Debugable, Serializable, MemoryWatchable {

    public synchronized long getMaxMemoryUsageInBytes() {
        return maxMemoryUsageBytes;
    }

    private long maxMemoryUsageBytes = -1;
    private long count = 0;
    private double sqDiffFromMean = 0;
    private BigDecimal bigSqDiffFromMean = null;
    private double mean = 0;
    private long garbageCollectionTimeInMillis = 0;
    private boolean overflowed = false; // to handle overflow of the sq sum of diff from mean (can get reallllyyy
    // big)

    @Override public boolean enableAnyway() {
        if(super.enableAnyway() && !isEmittersSetup()) {
            setupEmitters();
            return true;
        }
        return false;
    }

    private void setupEmitters() {
        emitters = new ArrayList<>();
        // garbage collector for old and young gen
        List<GarbageCollectorMXBean> garbageCollectorBeans = java.lang.management.ManagementFactory.getGarbageCollectorMXBeans();
        for (GarbageCollectorMXBean garbageCollectorBean : garbageCollectorBeans) {
            if(debug) System.out.println("Setting up listener for gc: " + garbageCollectorBean); // todo change
            // to log
            // listen to notification from the emitter
            NotificationEmitter emitter = (NotificationEmitter) garbageCollectorBean;
            emitters.add(emitter);
            emitter.addNotificationListener(listener, null, null);
        }
    }

    private void tearDownEmitters() {
        if(emitters != null) {
            for(NotificationEmitter emitter : emitters) {
                try {
                    emitter.removeNotificationListener(listener);
                } catch (ListenerNotFoundException e) {
                    throw new IllegalStateException("already removed somehow...");
                }
            }
        }
    }

    private boolean isEmittersSetup() {
        return emitters != null;
    }

    private transient List<NotificationEmitter> emitters;

    private void readObject(ObjectInputStream in) throws ClassNotFoundException, IOException
    {
        in.defaultReadObject();
        setupEmitters();
    }

    private final NotificationListener listener = (notification, handback) -> {
        synchronized(MemoryWatcher.this) {
            if(isEnabled()) {
                if (notification.getType().equals(GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION)) {
                    GarbageCollectionNotificationInfo info = GarbageCollectionNotificationInfo.from((CompositeData) notification.getUserData());
                    long duration = info.getGcInfo().getDuration();
                    garbageCollectionTimeInMillis += duration;
//            String action = info.getGcAction();
//            GcInfo gcInfo = info.getGcInfo();
//            long id = gcInfo.getId();
                    Map<String, MemoryUsage> memoryUsageInfo = info.getGcInfo().getMemoryUsageAfterGc();
                    for (Map.Entry<String, MemoryUsage> entry : memoryUsageInfo.entrySet()) {
//                String name = entry.getKey();
                        MemoryUsage memoryUsageSnapshot = entry.getValue();
//                long initMemory = memoryUsage.getInit();
//                long committedMemory = memoryUsage.getCommitted();
//                long maxMemory = memoryUsage.getMax();
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

    private void handleDeltaNonOverflow(double delta) {
        double prev = sqDiffFromMean;
        sqDiffFromMean += Math.pow(delta, 2);
        if(sqDiffFromMean < prev) {
            overflow(delta);
        }
    }

    private void overflow(double delta) {
        overflowed = true;
        bigSqDiffFromMean = BigDecimal.valueOf(sqDiffFromMean);
        handleDeltaOverflow(delta);
    }

    private void handleDeltaOverflow(double delta) {
        BigDecimal bigDelta = BigDecimal.valueOf(delta);
        bigSqDiffFromMean = bigSqDiffFromMean.add(bigDelta.multiply(bigDelta));
    }

    private void addMemoryUsageReadingInBytesUnchecked(long usage) {
        maxMemoryUsageBytes = Math.max(maxMemoryUsageBytes, usage);
        // Welford's online algo for mean and variance
        count++;
        double delta = (double) usage - mean;
        mean += delta / count;
        if(overflowed) {
            handleDeltaOverflow(delta);
        } else {
            handleDeltaNonOverflow(delta);
        }
    }

    private void addMemoryUsageReadingInBytes(long usage) {
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

    public synchronized BigDecimal getBigVarianceMemoryUsageInBytes() {
        return bigSqDiffFromMean.divide(BigDecimal.valueOf(count), BigDecimal.ROUND_UP);
    }

    public synchronized double getVarianceMemoryUsageInBytes() {
        if(count == 0) {
            return -1;
        } else if(overflowed) {
            return getBigVarianceMemoryUsageInBytes().doubleValue();
        } else {
            return sqDiffFromMean / count; // population variance as we see all the readings of memory usage;
        }
    }

    public long getMemoryReadingCount() {
        return count;
    }

    public double getStdDevMemoryUsageInBytes() {
        if(overflowed) {
            return StatisticalUtilities.sqrt(getBigVarianceMemoryUsageInBytes()).doubleValue();
        }
        return Math.sqrt(getVarianceMemoryUsageInBytes());
    }

    @Override
    public synchronized boolean isDebug() {
        return debug;
    }

    private boolean debug = false;

    @Override
    public synchronized void setDebug(boolean state) {
        debug = state;
    }

    public synchronized long getGarbageCollectionTimeInMillis() {
        return garbageCollectionTimeInMillis;
    }

    @Override public String toString() {
        return "MemoryWatcher{" +
            super.toString() +
            ", maxMemoryUsageBytes=" + maxMemoryUsageBytes +
            '}';
    }

    public void reset() {
        count = 0;
        mean = 0;
        sqDiffFromMean = 0;
        garbageCollectionTimeInMillis = 0;
        maxMemoryUsageBytes = -1;
        bigSqDiffFromMean = null;
        overflowed = false;
    }

    public static void main(String[] args) {
        StopWatch stopWatch = new StopWatch();
        MemoryWatcher realMemWatcher = new MemoryWatcher();
        realMemWatcher.enable();
        MemoryWatcher memoryWatcher = new MemoryWatcher();
        stopWatch.enable();
        Random rand = new Random(0);
        memoryWatcher.overflow(0);
        int max = 1_000_000;
        for(int i = 0; i < max; i++) {
            memoryWatcher.addMemoryUsageReadingInBytesUnchecked(Math.abs(rand.nextInt(100)));
            if(rand.nextInt(max) < 10_000) {
                System.gc();
            }
        }
        stopWatch.disable();
        realMemWatcher.disable();
        System.out.println(realMemWatcher.getMaxMemoryUsageInBytes());
        System.out.println(realMemWatcher.getMeanMemoryUsageInBytes());
        System.out.println(realMemWatcher.getStdDevMemoryUsageInBytes());
        System.out.println(realMemWatcher.getGarbageCollectionTimeInMillis());
        System.out.println("----");
        System.out.println(stopWatch.getTimeNanos());
        System.out.println(TimeUnit.SECONDS.convert(stopWatch.getTimeNanos(), TimeUnit.NANOSECONDS));
    }
}
