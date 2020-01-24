package utilities;

import java.util.logging.Level;
import java.util.logging.Logger;

public class LogUtils {
    private LogUtils() {}

    public static Logger getLogger(Object object) {
        Logger logger = Logger.getLogger(object.getClass().getSimpleName() + "-" + object.hashCode());
        logger.setLevel(Level.OFF); // disable logs by default
        return logger;
    }

}
