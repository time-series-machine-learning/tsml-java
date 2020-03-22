package tsml.classifiers;

import java.io.File;

public interface Visualisable {
    default boolean setVisualisationSavePath(String path){
        File f = new File(path);
        boolean success=true;
        if(!f.isDirectory())
            success=f.mkdirs();
        return success;
    }

    void createVisualisation();
}
