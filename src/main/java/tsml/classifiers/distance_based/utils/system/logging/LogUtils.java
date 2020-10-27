package tsml.classifiers.distance_based.utils.system.logging;

import java.io.OutputStream;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.logging.*;

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.strings.StrUtils;

/**
 * Purpose: build loggers / handy logging functions
 * <p>
 * Contributors: goastler
 */
public class LogUtils {

    private LogUtils() {
    }

    public static final Logger DEFAULT_LOG = buildLogger(LogUtils.class);

    public static Logger updateLogLevel(Object src, Logger log, Level level) {
        // if setting a log level then this object needs its own logger instance to differentiate the logging levels.
        // i.e. everything by default is pointed at DEFAULT_LOG. If the level of the DEFAULT_LOG were to be changed it would affect every object's logging. Instead, a specific logger is required to house the log level for this specific object.
        if(log.equals(DEFAULT_LOG)) {
            // build the logger for this object. Only do this once if still using the DEFAULT_LOGGER. Once a bespoke logger has been created the log level can be mutated freely on that with no problems.
            log = LogUtils.buildLogger(src);
        }
        log.setLevel(level);
        return log;
    }

    public static Logger buildLogger(Object src) {
        String name;
        if(src instanceof Class) {
            name = ((Class) src).getSimpleName();
        } else if(src instanceof String) {
            name = (String) src;
        } else if(src instanceof EnhancedAbstractClassifier) {
            name = ((EnhancedAbstractClassifier) src).getClassifierName() + "_" + src.hashCode(); // todo bung this into interface
        } else {
            name = src.getClass().getSimpleName() + "_" + src.hashCode();
        }
        Logger logger = Logger.getLogger(name);
        Handler[] handlers = logger.getHandlers();
        for(Handler handler : handlers) {
            logger.removeHandler(handler);
        }
        logger.setLevel(Level.SEVERE); // disable all but severe error logs by default
        logger.addHandler(buildStdOutStreamHandler(new CustomLogFormat()));
        logger.addHandler(buildStdErrStreamHandler(new CustomLogFormat()));
        logger.setUseParentHandlers(false);
        return logger;
    }

    public static class CustomLogFormat extends Formatter {

        @Override
        public String format(final LogRecord logRecord) {
            String separator = " | ";
            //            return logRecord.getSequenceNumber() + separator +
            //                logRecord.getLevel() + separator +
            //                logRecord.getLoggerName() + separator +
            //                logRecord.getSourceClassName() + separator +
            //                logRecord.getSourceMethodName() + System.lineSeparator() +
            //                logRecord.getMessage() + System.lineSeparator();
            return LocalDateTime.now().toString().replace("T", " ") + separator + logRecord.getMessage() + System.lineSeparator();
        }
    }

    public static class CustomStreamHandler extends StreamHandler {

        public CustomStreamHandler(final OutputStream out, final Formatter formatter) {
            super(out, formatter);
        }

        @Override
        public synchronized void publish(LogRecord record) {
            super.publish(record);
            flush();
        }

        @Override
        public synchronized void close() throws SecurityException {
            flush();
        }
    }

    public static StreamHandler buildStdErrStreamHandler(Formatter formatter) {
        StreamHandler soh = new CustomStreamHandler(System.err, formatter);
        soh.setLevel(Level.SEVERE); //Default StdErr Setting
        return soh;
    }

    public static StreamHandler buildStdOutStreamHandler(Formatter formatter) {
        StreamHandler soh = new CustomStreamHandler(System.out, formatter);
        soh.setLevel(Level.ALL); //Default StdOut Setting
        return soh;
    }

    public static void logTimeContract(long timeNanos, long limitNanos, Logger logger, String name) {
        if(limitNanos > 0) {
            logger.fine(() -> {
                Duration limit = Duration.ofNanos(limitNanos);
                Duration time = Duration.ofNanos(timeNanos);
                Duration diff = limit.minus(time);
                return StrUtils.durationToHmsString(time) + " elapsed of " + StrUtils.durationToHmsString(limit) +
                               " " + name + " "
                               + "time "
                               + "limit, " + StrUtils.durationToHmsString(diff) + " train time remaining";
            });
        }
    }

}
