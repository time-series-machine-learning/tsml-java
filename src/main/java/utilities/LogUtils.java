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
        Handler[] handlers = logger.getHandlers();
        for(Handler handler : handlers) {
            logger.removeHandler(handler);
        }
        logger.setLevel(Level.OFF); // disable logs by default
        logger.addHandler(buildStdOutStreamHandler(new SimpleFormatter()));
        logger.addHandler(buildStdErrStreamHandler(new SimpleFormatter()));
        return logger;
    }

    public static StreamHandler buildStdErrStreamHandler(Formatter formatter) {
        StreamHandler soh = new StreamHandler(System.err, formatter);
        soh.setLevel(Level.SEVERE); //Default StdErr Setting
        return soh;
    }

    public static StreamHandler buildStdOutStreamHandler(Formatter formatter) {
        StreamHandler soh = new StreamHandler(System.out, formatter);
        soh.setLevel(Level.ALL); //Default StdOut Setting
        return soh;
    }

}
