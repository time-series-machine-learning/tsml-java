package tsml.classifiers.distance_based.utils;

/**
 * Purpose: enable / disable debugging.
 *
 * Contributors: goastler
 */
public interface Debugable {
    boolean isDebug();
    void setDebug(boolean state);
    default void enableDebug() {
        setDebug(true);
    }
    default void disableDebug() {
        setDebug(false);
    }

    default boolean getDebug() {
        return isDebug();
    }
}
