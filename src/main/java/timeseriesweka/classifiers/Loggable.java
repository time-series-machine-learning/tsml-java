package timeseriesweka.classifiers;

import java.util.logging.Logger;

public interface Loggable {
    Logger getLogger();
    void setLogger(Logger logger);
}
