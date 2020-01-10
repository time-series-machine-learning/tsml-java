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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MemoryWatcher extends Stated implements Debugable, Serializable, MemoryWatchable {

    public synchronized long getMaxMemoryUsageInBytes() {
        return maxMemoryUsageBytes;
    }

    private long maxMemoryUsageBytes = -1;
    private long size = 0;
    private long firstMemoryUsageReading;
    private long usageSum = 0;
    private long usageSumSq = 0;
    private long garbageCollectionTimeInMillis = 0;

    @Override public boolean enableAnyway() {
        boolean change = super.enableAnyway();
        if(change && !isEmittersSetup()) {
            setupEmitters();
        }
        return change;
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
                        addMemoryUsageReadingBytes(memoryUsage);
                    }
                }
            }
        }
    };

    public MemoryWatcher() {
        super();
        setupEmitters();
    }

    public MemoryWatcher(State state) {
        super(state);
    }

    private void addMemoryUsageReadingBytes(long usage) {
        if(isEnabled()) {
            if(usage > maxMemoryUsageBytes) {
                maxMemoryUsageBytes = usage;
            }
            // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
            if(size == 0) {
                firstMemoryUsageReading = usage;
            }
            size++;
            long diff = usage - firstMemoryUsageReading;
            usageSum += diff;
            usageSumSq += Math.pow(diff, 2);
        }
    }

    public synchronized long getMeanMemoryUsageInBytes() {
        return firstMemoryUsageReading + usageSum / size;
    }

    public synchronized long getVarianceMemoryUsageInBytes() {
        return (usageSumSq - (usageSum * usageSum) / size) / (size - 1);
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
}
