package tsml.transformers;

import experiments.data.DatasetLoading;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import utilities.InstanceTools;
import weka.attributeSelection.PrincipalComponents;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;

/**
 * Transformer to generate Principal Components. Uses weka attribute selection PrincipalComponents. There is also
 * a weka Filter PrincipalComponents, but it is confusingly structured, like all Weka Filters, and has protected
 * methods. The down side of using the ArttributeSelection version is that it does not have the capability to set
 * a maximum number of attributes. This implementation creates a full transform then deletes attributes. This could
 * be wasteful in memory for large attribute spaces (although PCA will internally need mxm memory anyway.
 *
 * This assumes that the PrincipalComponents sorts the eignevectors so the first has most variance.
 * I'm 99.9% sure it does
 *
 * @author Tony Bagnall (ajb)
 */
public class PCA implements TrainableTransformer {

    private int numAttributesToKeep; //changed this to constructor as you could change number of atts to keep after fitting
    private PrincipalComponents pca;
    private boolean isFit = false;
    private ConstantAttributeRemover remover;

    public PCA(){
        this(100);
    }

    public PCA(int attsToKeep){
        pca = new PrincipalComponents();
        numAttributesToKeep = Math.max(1, attsToKeep);
        
        System.out.println(numAttributesToKeep);
    }

    @Override
    public void fit(Instances data){
        numAttributesToKeep = Math.min(data.numAttributes()-1, numAttributesToKeep);

        try{
            //Build the evaluator
            //this method is sets the names of the componenets used.
            pca.setMaximumAttributeNames(numAttributesToKeep); 
            pca.setVarianceCovered(1.0);
            pca.buildEvaluator(data);
            isFit = true;
        }catch(Exception e)
        {
            throw new RuntimeException(" Error in Transformers/PCA when fitting the PCA transform");
        }
    }

    @Override
    public boolean isFit(){
        return isFit;
    }

    @Override
    public Instances transform(Instances data) {

        if(!isFit)
            throw new RuntimeException("Fit PCA before transforming");

           

        Instances newData= null;
        try {
            newData = pca.transformedData(data);
            
            if(remover == null){
                remover = new ConstantAttributeRemover();
                remover.fit(newData);
            }

            newData = remover.transform(newData);
        } catch (Exception e) {
            throw new RuntimeException(" Error in Transformers/PCA when performing the PCA transform: "+e);
        }
        return newData;
    }

    @Override
    public Instance transform(Instance inst) {
        if(!isFit)
            throw new RuntimeException("Fit PCA before transforming");

        Instance newInst= null;
        try {
            newInst = pca.convertInstance(inst);

            /*for(int del:attsToRemove)
                newInst.deleteAttributeAt(del);*/
            //TODO: replace with Truncator
            while(newInst.numAttributes()-1>numAttributesToKeep)
               newInst.deleteAttributeAt(newInst.numAttributes()-2);

        } catch (Exception e) {
            e.printStackTrace();
        }

        return newInst;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        if(data.numAttributes()-1<numAttributesToKeep)
            numAttributesToKeep=data.numAttributes()-1;
        return null;
    }


    public static void main(String[] args) throws Exception {
        //Aarons local path for testing.
        String local_path = "D:\\Work\\Data\\Univariate_ts\\"; //Aarons local path for testing.
        //String m_local_path = "D:\\Work\\Data\\Multivariate_ts\\";
        //String m_local_path_orig = "D:\\Work\\Data\\Multivariate_arff\\";
        String dataset_name = "ChinaTown";


        Instances train = DatasetLoading.loadData(local_path + dataset_name + File.separator + dataset_name+"_TRAIN.ts");
        Instances test  = DatasetLoading.loadData(local_path + dataset_name + File.separator + dataset_name+"_TEST.ts");

        /*Instances train= DatasetLoading.loadData("Z:\\ArchiveData\\Univariate_arff\\Chinatown\\Chinatown_TRAIN.arff");
        Instances test= DatasetLoading.loadData("Z:\\ArchiveData\\Univariate_arff\\Chinatown\\Chinatown_TEST.arff");*/
        
        
        /*PCA pca=new PCA(1);
        pca.fit(train);
        Instances trans=pca.transform(train);
 //       Instances trans2=pca.transform(test);
        System.out.println(" Transfrom 1"+trans);
        System.out.println("Num attribvvutes = "+trans.numAttributes());*/
//        System.out.println(" Transfrom 2"+trans2);

        ShapeletTransformClassifier st = new ShapeletTransformClassifier();
        st.setPCA(true, 100);
        st.buildClassifier(train);
        double acc = utilities.ClassifierTools.accuracy(test, st);
        System.out.println("acc: " + acc);

    }
}
