package utilities;

import java.util.logging.*;

public class LogUtils {
    private LogUtils() {}

    public static Logger getLogger(Object object) {
        String name = object.getClass().getSimpleName();
        if(object instanceof Class) {
            name = ((Class) object).getSimpleName();
        } else {
            name = object.getClass().getSimpleName() + "-" + object.hashCode();
        }
        Logger logger = Logger.getLogger(name);
        logger.setLevel(Level.OFF); // disable logs by default
        logger.addHandler(buildStdOutStreamHandler(new SimpleFormatter()));
        logger.addHandler(buildStdErrStreamHandler(new SimpleFormatter()));
        logger.setUseParentHandlers(false);
        return logger;
    }

    public static StreamHandler buildStdErrStreamHandler(Formatter formatter) {
        StreamHandler soh = new StreamHandler(System.err, formatter) {
            @Override
            public synchronized void publish(LogRecord record) {
                super.publish(record);
                flush();
            }

            @Override
            public synchronized void close() throws SecurityException {
                flush();
            }
        };
        soh.setLevel(Level.SEVERE); //Default StdErr Setting
        return soh;
    }

    public static StreamHandler buildStdOutStreamHandler(Formatter formatter) {
        StreamHandler soh = new StreamHandler(System.out, formatter) {
            @Override
            public synchronized void publish(LogRecord record) {
                super.publish(record);
                flush();
            }

            @Override
            public synchronized void close() throws SecurityException {
                flush();
            }
        };
        soh.setLevel(Level.ALL); //Default StdOut Setting
        return soh;
    }

}
