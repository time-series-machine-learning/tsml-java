package tsml.classifiers;

import weka.core.Instance;

import java.io.File;

public interface Interpretable {

    boolean setInterpretabilitySavePath(String path);

    //output summary of how the last classifier prediction was made
    boolean lastClassifiedInterpretability() throws Exception;

    default boolean createInterpretabilityDirectories(String path){
        File f = new File(path);
        boolean success=true;
        if(!f.isDirectory())
            success=f.mkdirs();
        return success;
    }
}
