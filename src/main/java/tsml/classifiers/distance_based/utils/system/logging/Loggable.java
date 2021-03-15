package tsml.classifiers.distance_based.utils.system.logging;

import akka.event.Logging;

import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Handle different log levels for providing anywhere from none to verbose output via logging.
 *
 * Contributors: goastler
 */
public interface Loggable {
    default Level getLogLevel() {
        return getLogger().getLevel();
    }
    
    default void setLogLevel(Level level) {
        getLogger().setLevel(level);
    }
    
    default void setLogLevel(String level) {
        setLogLevel(Level.parse(level.toUpperCase()));
    }
    
    Logger getLogger();

    /**
     * Manually specify the logger to log to. This is helpful to share loggers between classes / insts.
     * @param logger
     */
    void setLogger(Logger logger);
}
