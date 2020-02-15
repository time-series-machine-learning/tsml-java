package utilities;

import tsml.classifiers.Checkpointable;

import java.io.*;
import java.util.logging.Logger;

public class CheckpointUtils {

    private CheckpointUtils() {}

    public static final String checkpointFileName = "checkpoint.ser";
    public static final String tempCheckpointFileName = checkpointFileName + ".tmp";

    public static boolean loadFromSingleCheckpoint(Checkpointable checkpointable, Logger logger) {
        if(!checkpointable.isCheckpointLoadingEnabled()) {
            logger.info("skipping loading checkpoint due to loading disabled");
            return false;
        }
        final String path = checkpointable.getSavePath() + checkpointFileName;
        logger.info(() -> "loading from checkpoint: " + path);
        try {
            checkpointable.loadFromFile(path);
            logger.info(() -> "loaded from checkpoint: " + path);
            checkpointable.setLastCheckpointTimeStamp(System.nanoTime());
            return true;
        } catch(Exception e) {
            logger.info("failed to load from checkpoint: " + e.getMessage());
        }
        return false;
    }


    public static boolean saveToSingleCheckpoint(Checkpointable checkpointable, Logger logger,
                                                 boolean ignoreInterval) throws Exception {
        if(!checkpointable.isCheckpointSavingEnabled()) {
            logger.info("skipping saving checkpoint due to loading disabled");
            return false;
        }
        if(!ignoreInterval) {
            if(!checkpointable.hasCheckpointIntervalElapsed()) {
                logger.info("skipping saving checkpoint as interval has not elapsed");
                return false;
            }
        } else {
            logger.info("ignoring checkpoint interval");
        }
        final String checkpointDirPath = checkpointable.getSavePath();
        final String tmpPath = checkpointDirPath + tempCheckpointFileName;
        final String path = checkpointDirPath + checkpointFileName;
        logger.info(() -> "saving checkpoint to: " + path);
        checkpointable.saveToFile(tmpPath);
        final boolean success = new File(tmpPath).renameTo(new File(path));
        if(!success) {
            throw new IllegalStateException("could not rename checkpoint file");
        } else {
            logger.info(() -> "saved checkpoint to: " + path);
            return true;
        }
    }

    public static boolean saveToSingleCheckpoint(Checkpointable checkpointable, Logger logger) throws Exception {

        return saveToSingleCheckpoint(checkpointable, logger, false);
    }

    public static void serialise(Object serializable, String path) throws Exception {
        try (FileUtils.FileLock fileLocker = new FileUtils.FileLock(path);
             FileOutputStream fos = new FileOutputStream(fileLocker.getFile());
             ObjectOutputStream out = new ObjectOutputStream(fos)) {
            out.writeObject(serializable);
        }
    }

    public static Object deserialise(String path) throws Exception{
        Object obj = null;
        try (FileUtils.FileLock fileLocker = new FileUtils.FileLock(path);
             FileInputStream fis = new FileInputStream(fileLocker.getFile());
             ObjectInputStream in = new ObjectInputStream(fis)) {
            obj = in.readObject();
        }
        return obj;
    }
}
