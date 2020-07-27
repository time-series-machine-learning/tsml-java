package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import utilities.FileUtils;

import java.io.*;
import java.nio.file.Files;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;

public class BaseCheckpointer implements Checkpointer {

    public static Logger DEFAULT_LOGGER = LogUtils.buildLogger(BaseCheckpointer.class);
    private static final long serialVersionUID = 1;
    private String checkpointDirPath;
    private String checkpointFileName;
    private long minCheckpointIntervalNanos;
    private long lastCheckpointTimeStamp = 0;
    private boolean load;
    // assume work has been done when no loading has occurred
    private boolean checkpointSinceLoad = true;
    private final Object target;

    public BaseCheckpointer(final Object target) {
        this.target = target;
        setLoad(true);
        setMinCheckpointIntervalNanos(TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS));
        setCheckpointFileName("checkpoint.ser.gz");
    }

    public Logger getLogger() {
        if(target instanceof Loggable) {
            return ((Loggable) target).getLogger();
        } else {
            return DEFAULT_LOGGER;
        }
    }

    public void setLogger(final Logger logger) {
        throw new UnsupportedOperationException();
    }

    public void saveToFile(String path) throws Exception {
        File tmp = new File(path + ".tmp");
        File main = new File(path);
        try (FileUtils.FileLock fileLocker = new FileUtils.FileLock(main);
                FileOutputStream fos = new FileOutputStream(tmp);
                GZIPOutputStream gos = new GZIPOutputStream(fos);
                ObjectOutputStream out = new ObjectOutputStream(gos)) {
            out.writeObject(target);
            Files.move(tmp.toPath(), main.toPath(), REPLACE_EXISTING);
            getLogger().info("saved checkpoint to " + path);
        }
    }

    public void loadFromFile(String path) throws Exception {
        Object obj = null;
        try (FileUtils.FileLock fileLocker = new FileUtils.FileLock(path);
                FileInputStream fis = new FileInputStream(fileLocker.getFile());
                GZIPInputStream gis = new GZIPInputStream(fis);
                ObjectInputStream in = new ObjectInputStream(gis)) {
            obj = in.readObject();
        }
        copyFromSerObject(obj);
        getLogger().info("loaded checkpoint from " + path);
    }

    @Override public boolean loadCheckpoint() throws Exception {
        final String checkpointFilePath = getCheckpointFilePath();
        if(isCheckpointPathSet() && load) {
            try {
                loadFromFile(checkpointFilePath);
                load = false;
                checkpointSinceLoad = false;
                return true;
            } catch(FileNotFoundException e) {
                getLogger().info("no checkpoint found at " + checkpointFilePath);
            } catch(Exception e) {
                getLogger().info("failed to load checkpoint from " + checkpointFilePath + " : " + e.toString());
            }
        }
        return false;
    }

    @Override public final void copyFromSerObject(final Object src) throws Exception {
        CopierUtils.shallowCopyFrom(src, target, CopierUtils.findSerialisableFields(src));
    }

    public String getCheckpointFilePath() {
        return checkpointDirPath + "/" + checkpointFileName;
    }

    @Override public boolean checkpointIfIntervalExpired() throws Exception {
        if(isCheckpointIntervalExpired() && isCheckpointPathSet()) {
            getLogger().info("checkpoint expired");
            return saveCheckpoint();
        }
        return false;
    }

    private boolean saveCheckpoint() throws Exception {
        final String checkpointFilePath = getCheckpointFilePath();
        try {
            saveToFile(checkpointFilePath);
            lastCheckpointTimeStamp = System.nanoTime();
            checkpointSinceLoad = true;
            return true;
        } catch(Exception e) {
            getLogger().info("failed to save checkpoint to " + checkpointFilePath + " : " + e.toString());
            throw e;
        }
    }

    @Override public boolean checkpoint() throws Exception {
        if(isCheckpointPathSet()) {
            getLogger().info("saving checkpoint");
            return saveCheckpoint();
        }
        return false;
    }

    @Override public boolean checkpointIfWorkDone() throws Exception {
        if(checkpointSinceLoad) {
            return checkpoint();
        }
        return false;
    }

    @Override public String getCheckpointFileName() {
        return checkpointFileName;
    }

    @Override public void setCheckpointFileName(final String checkpointFileName) {
        this.checkpointFileName = checkpointFileName;
    }

    @Override public String getCheckpointDirPath() {
        return checkpointDirPath;
    }

    @Override public void setCheckpointDirPath(final String checkpointDirPath) {
        this.checkpointDirPath = checkpointDirPath;
    }

    @Override public long getMinCheckpointIntervalNanos() {
        return minCheckpointIntervalNanos;
    }

    @Override public void setMinCheckpointIntervalNanos(final long minCheckpointIntervalNanos) {
        this.minCheckpointIntervalNanos = minCheckpointIntervalNanos;
    }

    @Override public boolean setCheckpointTimeHours(final int t) {
        setMinCheckpointIntervalNanos(t, TimeUnit.HOURS);
        return true;
    }

    @Override public long getLastCheckpointTimeStamp() {
        return lastCheckpointTimeStamp;
    }

    public void setLastCheckpointTimeStamp(final long lastCheckpointTimeStamp) {
        this.lastCheckpointTimeStamp = lastCheckpointTimeStamp;
    }

    @Override public boolean isLoad() {
        return load;
    }

    @Override public void setLoad(final boolean load) {
        this.load = load;
    }

}
