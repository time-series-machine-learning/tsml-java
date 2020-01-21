package utilities;

import java.util.logging.Level;
import java.util.logging.Logger;

public class LogUtils {
    private LogUtils() {}

    public static Logger getLogger(Object object) {
        return Logger.getLogger(object.getClass().getSimpleName() + "-" + object.hashCode());
    }

    public static boolean isAboveLevel(final Logger logger, final Level level) {
        return logger.getLevel().intValue() >= level.intValue(); // todo
    }
}
