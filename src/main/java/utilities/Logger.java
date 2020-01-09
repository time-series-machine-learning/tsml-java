package utilities;

import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.time.LocalDateTime;

public class Logger {

    private Writer writer;
    private final String tag;
    private boolean enabled = true;

    public Logger(final OutputStream destination, final String tag) {
        setDestination(destination);
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
                writer.write(log);
                writer.write(System.lineSeparator());
                writer.flush();
            } catch(IOException e) {
                throw new IllegalStateException(e);
            }
        }
    }

    public Writer getWriter() {
        return writer;
    }

    public String getTag() {
        return tag;
    }

    public Logger setDestination(OutputStream destination) {
        writer = new BufferedWriter(new OutputStreamWriter(destination));
        return this;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public Logger setEnabled(final boolean enabled) {
        this.enabled = enabled;
        return this;
    }
}
