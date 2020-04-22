package tsml.classifiers.distance_based.utils.checkpointing;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Set;
import java.util.function.Predicate;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import tsml.classifiers.Checkpointable;

import java.io.*;
import java.util.logging.Logger;
import tsml.classifiers.distance_based.utils.classifier_mixins.Copy;
import utilities.FileUtilities.FileLock;

/**
 * Purpose: utilities for checkpointing to a single file. This performs usual checks / logging along the way.
 *
 * Contributors: goastler
 */
public class CheckpointUtils {

    public static final Predicate<Field> TRANSIENT = field -> Modifier.isTransient(field.getModifiers());

    private CheckpointUtils() {}

    public static final String checkpointFileName = "checkpoint.ser";
    public static final String tempCheckpointFileName = checkpointFileName + ".tmp";

    /**
     * load from a single checkpoint
     * @param checkpointable the thing to be checkpointed
     * @param logger somewhere to log
     * @return
     * @throws Exception
     */
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

    /**
     * save to a single checkpoint
     * @param checkpointable the thing to be checkpointed
     * @param logger somewhere to log
     * @param ignoreInterval whether to ignore whether the checkpoint interval has elapsed
     * @return
     * @throws Exception
     */
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

    /**
     * serialise and compress
     * @param serializable
     * @param path
     * @throws Exception
     */
    public static void serialise(Object serializable, String path) throws Exception {
        try (FileLock fileLocker = new FileLock(path);
             FileOutputStream fos = new FileOutputStream(fileLocker.getFile());
             GZIPOutputStream gos = new GZIPOutputStream(fos);
             ObjectOutputStream out = new ObjectOutputStream(gos)) {
            out.writeObject(serializable);
        }
    }

    /**
     * deserialise and decompress
     * @param path
     * @return
     * @throws Exception
     */
    public static Object deserialise(String path) throws Exception{
        Object obj = null;
        try (FileLock fileLocker = new FileLock(path);
             FileInputStream fis = new FileInputStream(fileLocker.getFile());
             GZIPInputStream gis = new GZIPInputStream(fis);
             ObjectInputStream in = new ObjectInputStream(gis)) {
            obj = in.readObject();
        }
        return obj;
    }

    /**
     * find the fields which are not recorded in serialisation (as these don't need copying)
     * @param obj
     * @return
     */
    public static Set<Field> findSerFields(Object obj) {
        return Copy.findFields(obj.getClass(), TRANSIENT.negate().and(Copy.DEFAULT_FIELDS));
    }
}
