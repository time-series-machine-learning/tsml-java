package tsml.classifiers.distance_based.utils.system.memory;

/**
 * Purpose: get stats related to memory.
 *
 * Contributors: goastler
 */
public interface MemoryWatchable {
    long getMaxMemoryUsage();
    
    
    static void gc() {
        // do it twice and this automagically cleans up memory somehow...
        System.gc(); 
        System.gc();
        // above may have put some objs in a queue for finalization, so let's clear them out
        System.runFinalization();
        System.runFinalization();
    }
}
