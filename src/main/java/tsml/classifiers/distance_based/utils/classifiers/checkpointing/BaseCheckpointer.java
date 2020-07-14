package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import tsml.classifiers.distance_based.utils.classifiers.Copier;
import utilities.FileUtils;

import java.io.*;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class BaseCheckpointer implements Checkpointer {
    private String checkpointDirPath;
    private String checkpointFileName = "checkpoint.ser.gz";
    private long minCheckpointIntervalNanos = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    private long lastCheckpointTimeStamp = 0;
    private boolean loadCheckpoint;
    private final Copier target;

    public BaseCheckpointer(final Copier target) {
        this.target = target;
        setLoadCheckpoint(true);
    }

    public void saveToFile(String filename) throws Exception {
        try (FileUtils.FileLock fileLocker = new FileUtils.FileLock(filename);
                FileOutputStream fos = new FileOutputStream(fileLocker.getFile());
                GZIPOutputStream gos = new GZIPOutputStream(fos);
                ObjectOutputStream out = new ObjectOutputStream(gos)) {
            out.writeObject(target);
        }
    }

    public void loadFromFile(String filename) throws Exception{
        Object obj = null;
        try (FileUtils.FileLock fileLocker = new FileUtils.FileLock(filename);
                FileInputStream fis = new FileInputStream(fileLocker.getFile());
                GZIPInputStream gis = new GZIPInputStream(fis);
                ObjectInputStream in = new ObjectInputStream(gis)) {
            obj = in.readObject();
        }
        if(obj != null) {
            copyFromSerObject(obj);
        }
    }

    @Override public boolean loadCheckpoint() throws Exception {
        if(isCheckpointing()) {
            if(loadCheckpoint) {
                try {
                    loadFromFile(getCheckpointFilePath());
                    loadCheckpoint = false;
                    return true;
                } catch(FileNotFoundException ignored) {

                }
            }
        }
        return false;
    }

    @Override public final void copyFromSerObject(final Object obj) throws Exception {
        target.shallowCopyFrom(obj, CheckpointUtils.findSerFields(obj));
    }

    @Override public boolean saveCheckpoint(boolean force) throws Exception {
        if(isCheckpointing()) {
            if(force || isCheckpointIntervalExpired()) {
                saveToFile(getCheckpointFilePath());
                lastCheckpointTimeStamp = System.nanoTime();
                return true;
            }
        }
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
