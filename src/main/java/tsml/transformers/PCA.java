package tsml.transformers;

import experiments.data.DatasetLoading;
import weka.attributeSelection.PrincipalComponents;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

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
public class PCA implements Transformer {

//    PrincipalComponents pc=new weka.filters.unsupervised.attribute.PrincipalComponents();
    int numAttributesToKeep=100;
    weka.attributeSelection.PrincipalComponents pca=new PrincipalComponents();

    public void setNumAttributesToKeep(int a){
        numAttributesToKeep=a;
    }
    @Override
    public void fit(Instances data){
        if(data.numAttributes()-1<numAttributesToKeep)
            numAttributesToKeep=data.numAttributes()-1;
        try{
//Build the evaluator
            pca.setVarianceCovered(1.0);
            pca.buildEvaluator(data);
        }catch(Exception e)
        {
            throw new RuntimeException(" Error in Transformers/PCA when fitting the PCA transform");
        }
    }

    @Override
    public Instances transform(Instances data) {

        Instances newData= null;
        try {
            newData = pca.transformedData(data);
            while(newData.numAttributes()-1>numAttributesToKeep)
                newData.deleteAttributeAt(newData.numAttributes()-2);
        } catch (Exception e) {
            throw new RuntimeException(" Error in Transformers/PCA when performing the PCA transform: "+e);
        }
        return newData;
    }

    @Override
    public Instance transform(Instance inst) {
        Instance newInst= null;
        try {
            newInst = pca.convertInstance(inst);
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


    public static void main(String[] args) {
        Instances train= DatasetLoading.loadData("Z:\\ArchiveData\\Univariate_arff\\Chinatown\\Chinatown_TRAIN.arff");
        Instances test= DatasetLoading.loadData("Z:\\ArchiveData\\Univariate_arff\\Chinatown\\Chinatown_TEST.arff");
        PCA pca=new PCA();
        pca.fit(train);
        Instances trans=pca.transform(train);
 //       Instances trans2=pca.transform(test);
        System.out.println(" Transfrom 1"+trans);
        System.out.println("Num attribvvutes = "+trans.numAttributes());
//        System.out.println(" Transfrom 2"+trans2);

    }
}
