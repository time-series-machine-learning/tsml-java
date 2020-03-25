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

    void createVisualisation() throws Exception;


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
