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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class MemoryWatcher implements Debugable, Serializable, MemoryWatchable {

    public synchronized long getMaxMemoryUsageInBytes() {
        return maxMemoryUsageBytes;
    }

    private long maxMemoryUsageBytes = -1;
    private long size = 0;
    private long firstMemoryUsageReading;
    private long usageSum = 0;
    private long usageSumSq = 0;
    private long garbageCollectionTimeInMillis = 0;
    public enum State {
        DISABLED {
            @Override public void set(MemoryWatcher memoryWatcher) {
                if(memoryWatcher.state.ordinal() > State.DISABLED.ordinal()) {
                    memoryWatcher.pause();
                    if(memoryWatcher.emitters != null) {
                        for(NotificationEmitter emitter : memoryWatcher.emitters) {
                            try {
                                emitter.removeNotificationListener(memoryWatcher.listener);
                            } catch (ListenerNotFoundException e) {
                                throw new IllegalStateException("already removed somehow...");
                            }
                        }
                    }
                    memoryWatcher.state = State.DISABLED;
                }
            }
        },
        ENABLED {
            @Override public void set(final MemoryWatcher memoryWatcher) {
                if(memoryWatcher.state.ordinal() < State.ENABLED.ordinal()) {
                    memoryWatcher.emitters = new ArrayList<>();
                    // garbage collector for old and young gen
                    List<GarbageCollectorMXBean> garbageCollectorBeans = java.lang.management.ManagementFactory.getGarbageCollectorMXBeans();
                    for (GarbageCollectorMXBean garbageCollectorBean : garbageCollectorBeans) {
                        if(memoryWatcher.debug) System.out.println("Setting up listener for gc: " + garbageCollectorBean);
                        // listen to notification from the emitter
                        NotificationEmitter emitter = (NotificationEmitter) garbageCollectorBean;
                        memoryWatcher.emitters.add(emitter);
                        emitter.addNotificationListener(memoryWatcher.listener, null, null);
                    }
                    memoryWatcher.state = State.ENABLED;
                }
            }
        },
        RESUMED {
            @Override public void set(final MemoryWatcher memoryWatcher) {
                memoryWatcher.enable();
                memoryWatcher.state = State.RESUMED;
            }
        },
        PAUSED {
            @Override public void set(final MemoryWatcher memoryWatcher) {
                memoryWatcher.state = State.PAUSED;
            }
        };

        public abstract void set(MemoryWatcher memoryWatcher);
    }
    private State state = State.DISABLED;
    private transient List<NotificationEmitter> emitters;

    private void readObject(ObjectInputStream in) throws ClassNotFoundException, IOException
    {
        in.defaultReadObject();
        State orig = state;
        setState(State.DISABLED);
        setState(orig);
    }

    public void setState(State state) {
        state.set(this);
    }

    private final NotificationListener listener = (notification, handback) -> {
        synchronized(MemoryWatcher.this) {
            if(state.equals(State.RESUMED)) {
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
        enable();
    }

    public MemoryWatcher(boolean start) {
        this();
        if(start) {
            resume();
        }
    }

    private void addMemoryUsageReadingBytes(long usage) {
        if(state.equals(State.RESUMED)) {
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

    public synchronized void enable() {
        State.ENABLED.set(this);
    }

    public synchronized void disable() {
        State.DISABLED.set(this);
    }

    public void resume() {
        State.RESUMED.set(this);
    }

    public void pause() {
        State.PAUSED.set(this);
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
}
