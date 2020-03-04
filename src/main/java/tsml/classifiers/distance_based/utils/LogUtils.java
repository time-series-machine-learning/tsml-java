package tsml.classifiers.distance_based.utils;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.logging.*;

/**
 * Purpose: build loggers / handy logging functions
 *
 * Contributors: goastler
 */
public class LogUtils {
    private LogUtils() {}

    public static Logger getLogger(Object object) {
        String name;
        if(object instanceof Class) {
            name = ((Class) object).getSimpleName();
        } else if(object instanceof String) {
            name = (String) object;
        } else {
            name = object.getClass().getSimpleName() + "-" + object.hashCode();
        }
        Logger logger = Logger.getLogger(name);
        logger.setLevel(Level.SEVERE); // disable all but severe error logs by default
        logger.addHandler(buildStdOutStreamHandler(new CustomLogFormat()));
        logger.addHandler(buildStdErrStreamHandler(new CustomLogFormat()));
        logger.setUseParentHandlers(false);
        return logger;
    }

    public static class CustomLogFormat extends Formatter {

        @Override public String format(final LogRecord logRecord) {
            String separator = " | ";
            return logRecord.getSequenceNumber() + separator +
                logRecord.getLevel() + separator +
                logRecord.getLoggerName() + separator +
//                logRecord.getSourceClassName() + separator +
                logRecord.getSourceMethodName() + separator +
                logRecord.getMessage() + System.lineSeparator();
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

}
