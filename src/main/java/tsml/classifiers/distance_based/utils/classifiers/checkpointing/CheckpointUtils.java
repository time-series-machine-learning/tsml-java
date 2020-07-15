package tsml.classifiers.distance_based.utils.classifiers.checkpointing;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Set;
import java.util.function.Predicate;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import tsml.classifiers.Checkpointable;

import java.io.*;
import java.util.logging.Logger;

import tsml.classifiers.distance_based.utils.classifiers.Copier;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import utilities.FileUtils.FileLock;

/**
 * Purpose: utilities for checkpointing to a single file. This performs usual checks / logging along the way.
 *
 * Contributors: goastler
 */
public class CheckpointUtils {

    public static final Predicate<Field> TRANSIENT = field -> Modifier.isTransient(field.getModifiers());

    private CheckpointUtils() {}

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
        return CopierUtils.findFields(obj.getClass(), TRANSIENT.negate().and(CopierUtils.DEFAULT_FIELDS));
    }
}
