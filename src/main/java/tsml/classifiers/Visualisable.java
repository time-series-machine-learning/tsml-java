package tsml.classifiers;

import java.io.File;

/**
 * Interface for classifiers that can output visualisations of the final model.
 *
 * @author Matthew Middlehurst
 **/
public interface Visualisable {

    /**
     * Stores a path to save visualisation files to.
     *
     * @param path String directory path
     * @return true if path is valid, false otherwise.
     */
    boolean setVisualisationSavePath(String path);

    /**
     * Create model visualisations and save them to a set path.
     *
     * @return true if successful, false otherwise
     * @throws Exception if failure to set path or create visualisation
     */
    boolean createVisualisation() throws Exception;

    /**
     * Create a directory at a given path.
     *
     * @param path String directory path
     * @return true if folder is created successfully, false otherwise
     */
    default boolean createVisualisationDirectories(String path){
        File f = new File(path);
        boolean success=true;
        if(!f.isDirectory())
            success=f.mkdirs();
        return success;
    }
}