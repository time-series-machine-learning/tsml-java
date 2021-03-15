package tsml.classifiers.distance_based.utils.system.memory;

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
import utilities.Utilities;

/**
 * Purpose: watch the memory whilst enabled, tracking the mean, std dev, count, gc time and max mem usage.
 *
 * Note, most methods in this class are synchronized as the garbage collection updates come from another thread,
 * therefore all memory updates come from another thread and must be synced.
 *
 * Contributors: goastler
 */
public class MemoryWatcher extends Stated implements MemoryWatchable {

    public static void main(String[] args) {
        final MemoryWatcher memoryWatcher = new MemoryWatcher();
        memoryWatcher.start();
        final LinkedList<double[]> list = new LinkedList<>();
        int i = 0;
        while(true) {
            i++;
            Utilities.sleep(1);
            list.add(new double[1000]);
//            System.out.println(list.size());
            if(i % 10 == 0) {
                list.remove(0);
            }
            if(i % 10000 == 0) {
                System.out.println(memoryWatcher.getMaxMemoryUsage());
            }
        }
    }
    
    public synchronized void update() {
        // deliberately update memory usage using the used memory AT THE TIME OF INVOCATION. I.e. not necessarily the time of max memory usage!
        // we do this to work around the cases where the gc hasn't run, therefore we don't know the max memory over time
        // instead, we poll the memory usage at the current time
        if(maxMemoryUsage < 0) {
            maxMemoryUsage = Math.max(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory(), maxMemoryUsage);
        }
    }
    
    public synchronized long getMaxMemoryUsage() {
        update();
        return maxMemoryUsage;
    }

    private long maxMemoryUsage = -1;
    private transient NotificationListener listener = this::handleNotification;
    private boolean activeListener = false;
    
    public MemoryWatcher() {}

    @Override public void start() {
        addListener();
        super.start();
    }

    @Override public void stop() {
        MemoryWatchable.gc(); // clean up memory before stopping
        removeListener();
        super.stop();
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
        
        if(activeListener) throw new IllegalStateException("listener already active");
        activeListener = true;
    }
    
    private void removeListener() {
        // emitters are used to listen to each memory pool (usually young / old gen).
        // garbage collector for old and young gen
        List<GarbageCollectorMXBean> garbageCollectorBeans = java.lang.management.ManagementFactory.getGarbageCollectorMXBeans();
        for (GarbageCollectorMXBean garbageCollectorBean : garbageCollectorBeans) {
            // to log
            // listen to notification from the emitter
            NotificationEmitter emitter = (NotificationEmitter) garbageCollectorBean;
            try {
                emitter.removeNotificationListener(listener);
            } catch(ListenerNotFoundException ignored) {
                // nevermind, already been removed
                System.out.println("failed to remove listener");
            }
        }

        if(!activeListener) throw new IllegalStateException("listener already inactive");
        activeListener = false;
    }
    
    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
        // default deserialization
        ois.defaultReadObject();
        
        // stop if already started. Any memory watcher read from serialization should default to being stopped, like StopWatch
        if(isStarted()) {
            super.stop();
        }

    }

    private synchronized void handleNotification(final Notification notification, final Object handback) {
        if(notification.getType()
                   .equals(GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION)) {
            GarbageCollectionNotificationInfo info =
                    GarbageCollectionNotificationInfo
                            .from((CompositeData) notification.getUserData());
            for(Map.Entry<String, MemoryUsage> entry : info.getGcInfo().getMemoryUsageAfterGc().entrySet()) {
                MemoryUsage memoryUsageSnapshot = entry.getValue();
                long memoryUsage = memoryUsageSnapshot.getUsed();
                maxMemoryUsage = Math.max(memoryUsage, maxMemoryUsage);
            }
            for(Map.Entry<String, MemoryUsage> entry : info.getGcInfo().getMemoryUsageAfterGc().entrySet()) {
                MemoryUsage memoryUsageSnapshot = entry.getValue();
                long memoryUsage = memoryUsageSnapshot.getUsed();
                maxMemoryUsage = Math.max(memoryUsage, maxMemoryUsage);
            }
        }
    }
    
    @Override
    public String toString() {
        return "maxMemory: " + getMaxMemoryUsage();
    }

    public synchronized void onReset() {
        maxMemoryUsage = -1;
    }

}
