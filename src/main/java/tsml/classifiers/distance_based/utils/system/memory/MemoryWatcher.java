package tsml.classifiers.distance_based.utils.system.memory;

import com.google.common.testing.GcFinalization;
import com.sun.management.GarbageCollectionNotificationInfo;

import javax.management.ListenerNotFoundException;
import javax.management.Notification;
import javax.management.NotificationEmitter;
import javax.management.NotificationListener;
import javax.management.openmbean.CompositeData;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.MemoryUsage;
import java.util.*;

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

    public synchronized long getMaxMemoryUsage() {
        return maxMemoryUsage;
    }

    private long maxMemoryUsage = -1;
    private transient NotificationListener listener = this::handleNotification;

    public MemoryWatcher() {
        
    }

    private void addListener() {
        // emitters are used to listen to each memory pool (usually young / old gen).
        // garbage collector for old and young gen
        listener = this::handleNotification;
        List<GarbageCollectorMXBean> garbageCollectorBeans = java.lang.management.ManagementFactory.getGarbageCollectorMXBeans();
        for (GarbageCollectorMXBean garbageCollectorBean : garbageCollectorBeans) {
            // to log
            // listen to notification from the emitter
            NotificationEmitter emitter = (NotificationEmitter) garbageCollectorBean;
            /**
             * the memory update listener
             */
            emitter.addNotificationListener(listener, null, null);
        }
    }
    
    private void removeListener() throws ListenerNotFoundException {
        // emitters are used to listen to each memory pool (usually young / old gen).
        // garbage collector for old and young gen
        List<GarbageCollectorMXBean> garbageCollectorBeans = java.lang.management.ManagementFactory.getGarbageCollectorMXBeans();
        for (GarbageCollectorMXBean garbageCollectorBean : garbageCollectorBeans) {
            // to log
            // listen to notification from the emitter
            NotificationEmitter emitter = (NotificationEmitter) garbageCollectorBean;
            emitter.removeNotificationListener(listener);
        }
    }
    
    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
        // default deserialization
        ois.defaultReadObject();
        
        // stop if already started. Any memory watcher read from serialization should default to being stopped, like StopWatch
        if(isStarted()) {
            super.stop();
        }
        
        // when loading from serialisation, the listener is not preserved, therefore need to listen to memory again.
        addListener();

    }

    @Override protected void finalize() throws Throwable {
        super.finalize();
        removeListener();
    }

    private synchronized void handleNotification(final Notification notification, final Object handback) {
        if(MemoryWatcher.this.isStarted()) {
            if(notification.getType()
                       .equals(GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION)) {
                GarbageCollectionNotificationInfo info =
                        GarbageCollectionNotificationInfo
                                .from((CompositeData) notification.getUserData());
                Map<String, MemoryUsage> memoryUsageInfo = info.getGcInfo().getMemoryUsageAfterGc();
                for(Map.Entry<String, MemoryUsage> entry : memoryUsageInfo.entrySet()) {
                    MemoryUsage memoryUsageSnapshot = entry.getValue();
                    long memoryUsage = memoryUsageSnapshot.getUsed();
                    maxMemoryUsage = Math.max(memoryUsage, maxMemoryUsage);
                }
            }
        }
    }

    public synchronized void checkStopped() {
        super.checkStopped();
    }

    public synchronized void checkStarted() {
        super.checkStarted();
    }

    @Override public synchronized void start() {
        super.start();
    }

    @Override public synchronized boolean isStopped() {
        return super.isStopped();
    }

    @Override public synchronized void stop() {
        super.stop();
    }

    @Override public synchronized void resetAndStart() {
        super.resetAndStart();
    }

    @Override public synchronized void stopAndReset() {
        super.stopAndReset();
    }

    @Override public synchronized void optionalStart() {
        super.optionalStart();
    }

    @Override public synchronized void optionalStop() {
        super.optionalStop();
    }

    public synchronized boolean isStarted() {
        return super.isStarted();
    }

    @Override
    public String toString() {
        return "maxMemory: " + maxMemoryUsage;
    }

    @Override
    public synchronized void reset() {
        super.reset();
    }

    public synchronized void onReset() {
        maxMemoryUsage = 0;
    }

    /**
     * this cleans the memory by forcing the garbage collector invocation. Guava's finalization tools not only invoke
     * the gc but also create a weak ref and continue to invoke the gc until said weak ref is cleaned up. The theory
     * here is if the most recently created weak ref has been cleaned up, all older memory should have already been
     * dealt with / cleaned up as matter of priority over the latest memory allocation.
     */
    public void cleanup() {
        GcFinalization.awaitFullGc();
    }

}
