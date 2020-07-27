package tsml.classifiers.distance_based.utils.system.logging;

import java.util.logging.Logger;

/**
 * Purpose: simple interface to get the logger from an implementing class. Each logger is bespoke to an instance of a
 * class, therefore only access to that logger is granted, for the purpose of logging, and setting is not allowed
 * (otherwise we'd be logging from a logger unspecific to our context).
 *
 * Contributors: goastler
 */
public interface Loggable {
    Logger getLogger();

    default void setLogger(Logger logger) {

    }
}
