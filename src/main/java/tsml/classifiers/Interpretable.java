package tsml.classifiers;

import java.io.File;

/**
 * Interface for classifiers that can output files or visualisations for each prediction made.
 *
 * @author Matthew Middlehurst
 */
public interface Interpretable {

    /**
     * Stores a path to save interpretability files to.
     *
     * @param path String directory path
     * @return true if path is valid, false otherwise.
     */
    boolean setInterpretabilitySavePath(String path);

    /**
     * Outputs a summary/visualisation of how the last classifier prediction was made to a set path
     *
     * @return true if successful, false otherwise
     * @throws Exception if failure to set path or output files.
     */
    boolean lastClassifiedInterpretability() throws Exception;

    /**
     * Get a unique indentifier for the last prediction made, used for filenames etc.
     *
     * @return int ID for the last prediction
     */
    int getPredID();

    /**
     * Create a directory at a given path.
     *
     * @param path String directory path
     * @return true if folder is created successfully, false otherwise
     */
    default boolean createInterpretabilityDirectories(String path){
        File f = new File(path);
        boolean success=true;
        if(!f.isDirectory())
            success=f.mkdirs();
        return success;
    }
}
