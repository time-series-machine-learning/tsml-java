package utilities;

import java.util.logging.*;

public class LogUtils {
    private LogUtils() {}

    public static Logger getLogger(Object object) {
        Logger logger = Logger.getLogger(object.getClass().getSimpleName() + "-" + object.hashCode());
        logger.setLevel(Level.OFF); // disable logs by default
        logger.addHandler(buildStdErrStreamHandler(new SimpleFormatter()));
        logger.addHandler(buildStdOutStreamHandler(new SimpleFormatter()));
        return logger;
    }

    public static StreamHandler buildStdErrStreamHandler(Formatter formatter) {
        StreamHandler soh = new StreamHandler(System.out, formatter);
        soh.setLevel(Level.ALL); //Default StdOut Setting
        return soh;
    }

    public static StreamHandler buildStdOutStreamHandler(Formatter formatter) {
        StreamHandler soh = new StreamHandler(System.out, formatter);
        soh.setLevel(Level.SEVERE); //Default StdErr Setting
        return soh;
    }

}
