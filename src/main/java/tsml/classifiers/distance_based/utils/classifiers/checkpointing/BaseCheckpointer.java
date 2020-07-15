package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import utilities.FileUtils;

import java.io.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class BaseCheckpointer implements Checkpointer {

    public static Logger DEFAULT_LOGGER = LogUtils.buildLogger(BaseCheckpointer.class);
    private static final long serialVersionUID = 1;
    private Logger logger;
    private String checkpointDirPath;
    private String checkpointFileName;
    private long minCheckpointIntervalNanos;
    private long lastCheckpointTimeStamp = 0;
    private boolean loadCheckpoint;
    private final Object target;

    public BaseCheckpointer(final Object target) {
        this.target = target;
        setLoadCheckpoint(true);
        setLogger(DEFAULT_LOGGER);
        setMinCheckpointIntervalNanos(TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS));
        setCheckpointFileName("checkpoint.ser.gz");
    }

    public Logger getLogger() {
        return logger;
    }

    public void setLogger(final Logger logger) {
        Assert.assertNotNull(logger);
        this.logger = logger;
    }

    public void saveToFile(String path) throws Exception {
        try (FileUtils.FileLock fileLocker = new FileUtils.FileLock(path);
                FileOutputStream fos = new FileOutputStream(fileLocker.getFile());
                GZIPOutputStream gos = new GZIPOutputStream(fos);
                ObjectOutputStream out = new ObjectOutputStream(gos)) {
            out.writeObject(target);
            logger.info("saved checkpoint to " + path);
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
        if(obj != null) {
            copyFromSerObject(obj);
            logger.info("loaded checkpoint from " + path);
        }
    }

    @Override public boolean loadCheckpoint() throws Exception {
        final String checkpointFilePath = getCheckpointFilePath();
        if(isCheckpointing()) {
            if(loadCheckpoint) {
                try {
                    loadFromFile(checkpointFilePath);
                    loadCheckpoint = false;
                    return true;
                } catch(Exception e) {
                    logger.severe("failed to load checkpoint from " + checkpointFilePath + " " + e.toString());
                    throw e;
                }
            }
        }
        logger.info("checkpoint loading disabled");
        return false;
    }

    @Override public final void copyFromSerObject(final Object src) throws Exception {
        CopierUtils.shallowCopyFrom(src, target, CheckpointUtils.findSerFields(src));
    }

    @Override public boolean saveCheckpoint(boolean force) throws Exception {
        if(isCheckpointing()) {
            final boolean expired = isCheckpointIntervalExpired();
            if(force) {
                logger.info("force saving checkpoint");
            } else if(expired) {
                logger.info("checkpoint expired, saving new checkpoint");
            }
            if(force || expired) {
                final String checkpointFilePath = getCheckpointFilePath();
                try {
                    saveToFile(checkpointFilePath);
                    lastCheckpointTimeStamp = System.nanoTime();
                    return true;
                } catch(Exception e) {
                    logger.severe("failed to save checkpoint to " + checkpointFilePath + " " + e.toString());
                    throw e;
                }
            } else {
                return false;
            }
        }
        logger.info("checkpoint saving disabled");
        return false;
    }

    public String getCheckpointFilePath() {
        return checkpointDirPath + "/" + checkpointFileName;
    }

    @Override public boolean saveCheckpoint() throws Exception {
        return saveCheckpoint(false);
    }

    @Override public boolean saveFinalCheckpoint() throws Exception {
        return saveCheckpoint(true);
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

    @Override public void setLastCheckpointTimeStamp(final long lastCheckpointTimeStamp) {
        this.lastCheckpointTimeStamp = lastCheckpointTimeStamp;
    }

    @Override public boolean isLoadCheckpoint() {
        return loadCheckpoint;
    }

    @Override public void setLoadCheckpoint(final boolean loadCheckpoint) {
        this.loadCheckpoint = loadCheckpoint;
    }
}
