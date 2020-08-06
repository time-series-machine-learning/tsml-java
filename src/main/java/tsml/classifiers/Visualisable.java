package tsml.classifiers;

import java.io.File;

/**
 * Interface for classifiers that can output visualisations of the final model.
 *
 * @author Matthew Middlehurst
 **/
public interface Visualisable {

    /**
     *
     *
     * @param path
     * @return
     */
    boolean setVisualisationSavePath(String path);

    /**
     *
     *
     * @return
     * @throws Exception
     */
    boolean createVisualisation() throws Exception;

    /**
     *
     *
     * @param path
     * @return
     */
    default boolean createVisualisationDirectories(String path){
        File f = new File(path);
        boolean success=true;
        if(!f.isDirectory())
            success=f.mkdirs();
        return success;
    }
}