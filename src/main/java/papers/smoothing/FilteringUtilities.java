
package papers.smoothing;

import java.io.File;
import java.io.FileFilter;
import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class FilteringUtilities {

    /**
     * Can't for the life of me make matlab2weka make the class value nominal when 
     * it saves the wekaOBJ. So the solution is just to edit the instance obj here and re-write
     * it after the fact... 
     */
    public static void fixClassValuesFromMatlab2Weka() throws Exception {
        String[] splits = { "TRAIN", "TEST" };
        
        String RAWdataPath = "Z:/Data/TSCProblems/";
        
//        String writePath = "Z:/Data/TSCProblems_FFT_zeroed_FCV/";
//        String dataPath = "Z:/Data/TSCProblems_FFT_zeroed/";
//        String writePath = "Z:/Data/TSCProblems_FFT_truncate_FCV/";
//        String dataPath = "Z:/Data/TSCProblems_FFT_truncate/";
//        String writePath = "Z:/Data/TSCProblems_PCA_smoothed_FCV/";
//        String dataPath = "Z:/Data/TSCProblems_PCA_smoothed/";
//        String writePath = "Z:/Data/TSCProblems_MovingAverage_FCV/";
//        String dataPath = "Z:/Data/TSCProblems_MovingAverage/";
//        String writePath = "Z:/Data/TSCProblems_Exponential_FCV/";
//        String dataPath = "Z:/Data/TSCProblems_Exponential/";
        String writePath = "Z:/Data/TSCProblems_Gaussian_FCV/";
        String dataPath = "Z:/Data/TSCProblems_Gaussian/";
        
//        String dset = "Adiac-DFT_1-";
        
        for (File dsetFile : (new File(dataPath)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isDirectory();
            }
        })) {
            String dset = dsetFile.getName();
            
            System.out.println("dset = " + dset);
            for (String split : splits) {
                String baseDset = dset.split("-")[0];
                Instances rawdata = ClassifierTools.loadData(RAWdataPath+baseDset+"/"+baseDset);

                Instances data = ClassifierTools.loadData(dataPath+dset+"/"+dset+"_"+split);
                Attribute rawClassAtt = rawdata.classAttribute();

                //can't remember why, but couldnt jsut use the class att for some reason
                //weka and dataset association or some bollocks
                FastVector<String> attVals = new FastVector();
                for (int i = 0; i < rawClassAtt.numValues(); i++)
                    attVals.add(rawClassAtt.value(i));
                Attribute newClassAtt = new Attribute(rawClassAtt.name(), attVals);

                
                //remove the old numeric att, and add the new nominal 
                double[] classVals = data.attributeToDoubleArray(data.classIndex());
                data.setClassIndex(0);
                data.deleteAttributeAt(data.numAttributes()-1);
                data.insertAttributeAt(newClassAtt, data.numAttributes());
                data.setClassIndex(data.numAttributes()-1);

                //set the vals for each inst 
                for (int i = 0; i < classVals.length; i++)
                    data.instance(i).setClassValue(classVals[i]);

                //and re-write
                ArffSaver saver = new ArffSaver();
                saver.setInstances(data);
                saver.setFile(new File(writePath+dset+"/"+dset+"_"+split+".arff"));
                saver.writeBatch();
            }
        }
    }
    
    public static void main(String[] args) throws Exception {
        fixClassValuesFromMatlab2Weka();
    }
}
