package tsml.classifiers.distance_based.utils.system.logging;

import akka.event.Logging;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Handle different log levels for providing anywhere from none to verbose output via logging.
 *
 * Contributors: goastler
 */
public interface Loggable {
    Level getLogLevel();
    
    void setLogLevel(Level level);
}
