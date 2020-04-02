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


    //Python debug code:

//    Process p = Runtime.getRuntime().exec("py pythonFile.py " + arguments);
//
//    BufferedReader out = new BufferedReader(new InputStreamReader(p.getInputStream()));
//    BufferedReader err = new BufferedReader(new InputStreamReader(p.getErrorStream()));
//
//    System.out.println("output : ");
//    String outLine = out.readLine();
//        while (outLine != null){
//        System.out.println(outLine);
//        outLine = out.readLine();
//    }
//
//    System.out.println("error : ");
//    String errLine = err.readLine();
//        while (errLine != null){
//        System.out.println(errLine);
//        errLine = err.readLine();
//    }
}