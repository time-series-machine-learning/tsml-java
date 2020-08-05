package tsml.classifiers;

import java.io.File;

/**
 * Interface for classifiers that can output visualisations of the final model.
 *
 * Author: Matthew Middlehurst
 **/
public interface Visualisable {

    boolean setVisualisationSavePath(String path);

    boolean createVisualisation() throws Exception;

    default boolean createVisualisationDirectories(String path){
        File f = new File(path);
        boolean success=true;
        if(!f.isDirectory())
            success=f.mkdirs();
        return success;
    }
}