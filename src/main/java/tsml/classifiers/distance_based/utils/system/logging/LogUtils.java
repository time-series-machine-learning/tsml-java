/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.utils.system.logging;

import java.io.OutputStream;
import java.time.LocalDateTime;
import java.util.concurrent.TimeUnit;
import java.util.logging.*;

import tsml.classifiers.distance_based.utils.experiment.TimeSpan;

/**
 * Purpose: build loggers / handy logging functions
 * <p>
 * Contributors: goastler
 */
public class LogUtils {

    private LogUtils() {
    }

    public static final Logger DEFAULT_LOG = getLogger("global");

    public static Logger updateLogLevel(Object src, Logger log, Level level) {
        // if setting a log level then this object needs its own logger instance to differentiate the logging levels.
        // i.e. everything by default is pointed at DEFAULT_LOG. If the level of the DEFAULT_LOG were to be changed it would affect every object's logging. Instead, a specific logger is required to house the log level for this specific object.
        if(DEFAULT_LOG.equals(log)) {
            // build the logger for this object. Only do this once if still using the DEFAULT_LOGGER. Once a bespoke logger has been created the log level can be mutated freely on that with no problems.
            log = LogUtils.getLogger(src);
        }
        log.setLevel(level);
        return log;
    }
    
    public static Logger getLogger(Object src) {
        final String name;
        
        if(src instanceof Class) {
            name = ((Class<?>) src).getCanonicalName();
        } else if(src instanceof String) {
            name = (String) src;
        } else {
            name = src.getClass().getCanonicalName();
        }
        
        Logger logger = Logger.getLogger(name);
        if(logger.getLevel() == null) {
            logger.setLevel(Level.SEVERE);
        }
        Handler[] handlers = logger.getHandlers();
        for(Handler handler : handlers) {
            logger.removeHandler(handler);
        }
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
    
    public static long logTimeContract(long timeNanos, long limitNanos, Logger logger, String name, long lastCallTimeStamp) {
        if(System.nanoTime() - lastCallTimeStamp > TimeUnit.NANOSECONDS.convert(10, TimeUnit.SECONDS)) {
            logTimeContract(timeNanos, limitNanos, logger, name);
            return System.nanoTime();
        } else {
            return lastCallTimeStamp;
        }
    }
    
    public static void logTimeContract(long timeNanos, long limitNanos, Logger logger, String name) {
        if(limitNanos > 0) {
            logger.info(() -> {
                TimeSpan limit = new TimeSpan(limitNanos);
                TimeSpan time = new TimeSpan(timeNanos);
                TimeSpan diff = new TimeSpan(limitNanos - timeNanos);
                return time.asTimeStamp() + " elapsed of " + limit.asTimeStamp() +
                               " " + name + " "
                               + "time "
                               + "limit, " + diff.asTimeStamp() + " remaining";
            });
        }
    }

}
