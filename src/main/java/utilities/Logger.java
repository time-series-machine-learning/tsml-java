package utilities;

import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Logger {

    private final Map<OutputStream, Writer> writers = new HashMap<>();
    private final String tag;
    private boolean enabled = true;

    public Logger(final OutputStream destination, final String tag) {
        addDestination(destination);
        this.tag = tag;
    }

    public Logger(final OutputStream destination, final Object caller) {
        this(destination, caller.getClass().getSimpleName() + "_" + caller.hashCode());
    }

    public Logger(final Object caller) {
        this(System.out, caller);
    }

    public Logger(final String tag) {
        this(System.out, tag);
    }

    public void log(Object... object) {
        if(enabled) {
            String log = LocalDateTime.now() + " | " + tag + " | " + StringUtils.join(object, " ");
            try {
                for(Writer writer : writers.values()) {
                    writer.write(log);
                    writer.write(System.lineSeparator());
                }
                for(Writer writer : writers.values()) {
                    writer.flush();
                }
            } catch(IOException e) {
                throw new IllegalStateException(e);
            }
        }
    }

    public String getTag() {
        return tag;
    }

    public boolean addDestination(OutputStream destination) {
        Writer writer = writers.get(destination);
        if(writer != null) {
            return false;
        } else {
            writers.put(destination, new BufferedWriter(new OutputStreamWriter(destination)));
            return true;
        }
    }

    public boolean removeDestination(OutputStream destination) {
        return writers.remove(destination) != null;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public Logger setEnabled(final boolean enabled) {
        this.enabled = enabled;
        return this;
    }
}
