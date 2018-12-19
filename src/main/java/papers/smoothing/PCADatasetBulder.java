
package papers.smoothing;

import java.io.File;
import utilities.ClassifierTools;
import weka.attributeSelection.PrincipalComponents;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

/**
 * Built the pca datasets in weka since I could do it quite easily. Many other filtered
 * version were simply done in matlab to same time and avoid potential mistakes in code
 * conversion/replication. In worst case, we can call matlab code from java to demonstrate
 * use case/reproducability.
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class PCADatasetBulder {

        
    public static void makePCAUCRDatasets() throws Exception {
        boolean transformBack = false;
        
        String[] dsets = development.DataSets.tscProblems85;
        assert(dsets.length == 85);
        
        int folds = 30;
        
        String readPath = "Z:/Data/TSCProblems/";
        String writePath;
        
        if (transformBack) 
            writePath = "Z:/Data/TSCProblems_PCA_smoothed/";
        else
            writePath = "Z:/Data/TSCProblems_PCA_comps/";
        
        double[] vars;
        
        if (transformBack)
            vars = new double[] { 0.9, 0.95, 0.99 }; //no point maintaing all variance in original space - jsut identical to original 
        else 
            vars = new double[] { 0.9, 0.95, 0.99, 1 }; 
        
        for (int varExpInd = 0; varExpInd < vars.length; varExpInd++) {
            double VARIANCE_EXPLAINED = vars[varExpInd];
            
            System.out.println(VARIANCE_EXPLAINED);
            
            for (int dset = 0; dset < dsets.length; dset++) {
                String filename = filenameBuild(dsets[dset], VARIANCE_EXPLAINED);
                (new File(writePath + filename + "/")).mkdir();

                System.out.println(filename);
                
                for (int f = 0; f < folds; f++) {
                    Instances train = ClassifierTools.loadData(readPath + dsets[dset] + "/" + dsets[dset] + "_TRAIN.arff");
                    Instances test = ClassifierTools.loadData(readPath + dsets[dset] + "/" + dsets[dset] + "_TEST.arff");

//                    System.out.println(train.toString());
                    
                    PrincipalComponents pca = new PrincipalComponents();
                    pca.setVarianceCovered(VARIANCE_EXPLAINED);
                    pca.setTransformBackToOriginal(transformBack);
                    pca.buildEvaluator(train);
                    Instances pcatrain = pca.transformedData(train);

                    Instances pcatest = new Instances(pca.transformedHeader());
                    for (Instance instance : test)
                        pcatest.add(pca.convertInstance(instance));

                    
                    //this was commented/the delete line inserted when overwriting the incorrect test split files
//                    ArffSaver saver = new ArffSaver();
//                    saver.setInstances(pcatrain);
//                    saver.setFile(new File(writePath + filename + "/" + filename + f + "_TRAIN.arff"));
//                    saver.writeBatch();
                    (new File(writePath + filename + "/" + filename + f + "_TEST.arff")).delete();

                    ArffSaver saver2 = new ArffSaver();
                    saver2.setInstances(pcatest);
                    saver2.setFile(new File(writePath + filename + "/" + filename + f + "_TEST.arff"));
                    saver2.writeBatch();
                }
            }
        }
    }
    
    /**
     * Want to avoid the decimal point in file/folder names, just to make sure everything 
     * plays along nicely across windows/linux etc
     */
    public static String filenameBuild(String dset, double PCAvar) { 
        String varStr = "";
        
        if (PCAvar == 1.0)
            varStr = "1";
        else {
            varStr = PCAvar + "";
            int oldLen = varStr.length();
            varStr = varStr.replace("0.", "");
            assert(varStr.length() == oldLen-2);
        } 
            
        return dset + "-PCA_" + varStr + "-";
    }
    
    public static double filenameGetPCAVar(String filename) { 
        String varStr = filename.split("-")[1].split("_")[1];
         //e.g Adiac-PCA_99-0_TEST => PCA_99 => 99
                
        if (varStr.equals("1"))
            return 1.0; 
        else {
            varStr = "0." + varStr;
            return Double.parseDouble(varStr);
        } 
    }
    
    public static String filenameGetDset(String filename) { 
        return filename.split("-")[0];
         //e.g Adiac-PCA_99-0_TEST => Adiac 
    }
    
    /**
     * Will of course only work for individual fold files, fold numbers not included in directory names
     */
    public static int filenameGetFold(String filename) { 
        return Integer.parseInt(filename.split("-")[2].split("_")[0]); 
        //e.g Adiac-PCA_99-0_TEST => 0_TEST => 0
    }
    
    public static void main(String[] args) throws Exception {
        makePCAUCRDatasets();
    }
}
