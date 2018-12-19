
package vector_classifiers;

import weka.classifiers.functions.PLSClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * Built for my (James') alcohol datasets, to allow comparative testing between TSC approaches 
 * and the de-facto chemometrics approach, Partial Least Squares regression
 * 
 * Extends the weka PLSClassifier, and essentially just converts the nominal class valued
 * dataset passed (initial intention being the ifr non-invasive whiskey datasets)
 * and does the standard regression, before converting the output back into a discrete class value
 * 
 * This version ignores the true values of the classes, instead just representing as class 0,1...c
 * If c>2, classes should be ordered (ascending or descending, doesn't matter) by whatever logical
 * ordering makes sense for the dataset. 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class PLSNominalClassifier extends PLSClassifier {

    protected Attribute classAttribute;
    protected int classind;
    protected int numClasses;
    
    public PLSNominalClassifier() {
        super();
    }
    
    public int getNumComponents() {
         return this.m_Filter.getNumComponents();
    }
    
    public void setNumComponents(int value) {
         this.m_Filter.setNumComponents(value);
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        Instances train = new Instances(data);
        
        numClasses = train.numClasses();
        classind = train.classIndex();
        classAttribute = train.classAttribute();
        
        FastVector<Attribute> atts = new FastVector<>(train.numAttributes());
        for (int i = 0; i < train.numAttributes(); i++) {
            if (i != classind)
                atts.add(train.attribute(i));
            else {
                //class attribute
                Attribute numericClassAtt = new Attribute(train.attribute(i).name());
                atts.add(numericClassAtt);
            }
        }
        
        Instances temp = new Instances(train.relationName(), atts, train.numInstances());
        temp.setClassIndex(classind);
        
        for (int i = 0; i < train.numInstances(); i++) {
            temp.add(new DenseInstance(1.0, train.instance(i).toDoubleArray()));
            temp.instance(i).setClassValue(train.instance(i).classValue());
        }
        
        train = temp;
        
        //datset is in the proper format, now do the model fitting as normal
        super.buildClassifier(train);
    }
    
    protected double convertNominalToNumeric(String strClassVal) {
        return Double.parseDouble(strClassVal.replaceAll("[A-Za-z ]", ""));
    }
    
    public double regressInstance(Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return utilities.GenericTools.indexOfMax(distributionForInstance(instance));
    }
    
    public double[] distributionForInstance(Instance instance) throws Exception {
        double regpred = regressInstance(instance);
        
        double[] dist = new double[numClasses];
        
        if (regpred <= 0)
            dist[0] = 1.0;
        else if (regpred >= numClasses-1)
            dist[numClasses-1] = 1.0;
        else {
            for (int i = 1; i < numClasses; i++) {
                if (regpred < i) {
                    double t = regpred % 1;
                    
                    dist[i] = t;
                    dist[i-1] = 1-t;
                    
                    break;
                }    
            }
        }
        
        return dist;
    }
    
}
