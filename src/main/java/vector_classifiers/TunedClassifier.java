
package vector_classifiers;

import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import evaluation.tuning.Tuner;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Given 
 *      1) a tuning method, 
 *      2) a base classifier with a well formed setOptions(String[]) method 
 *          (that must extend AbstractClassifier, the Classifier interface 
 *          alone does not provide setOptions()) 
 *      3) a parameter space expressed as keys (that correspond to flags in the setOptions() method) 
 *          to lists of values that these parameters can take.
 * 
 * This class will select the best parameter set on the given dataset according to the 
 * selection and evaluation methods described by the tuner, and build the base classifier
 * with the best parameters found
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class TunedClassifier extends AbstractClassifier {

    ParameterSpace space = null;
    Tuner tuner = null;
    AbstractClassifier classifier = null;

    
    public TunedClassifier() { 
    }
    
    public TunedClassifier(AbstractClassifier classifier, ParameterSpace space) { 
        this.classifier = classifier;
        this.space = space;
        this.tuner = new Tuner(); 
    }
    
    public TunedClassifier(AbstractClassifier classifier, ParameterSpace space, Tuner tuner) { 
        this.classifier = classifier;
        this.space = space;
        this.tuner = tuner;
    }
    
    public boolean getCloneClassifierForEachParameterEval() {
        return tuner.getCloneClassifierForEachParameterEval();
    }

    public void setCloneClassifierForEachParameterEval(boolean clone) {
        tuner.setCloneClassifierForEachParameterEval(clone);
    }
    
    public ParameterSpace getSpace() {
        return space;
    }

    public void setSpace(ParameterSpace space) {
        this.space = space;
    }

    public Tuner getTuner() {
        return tuner;
    }

    public void setTuner(Tuner tuner) {
        this.tuner = tuner;
    }

    public AbstractClassifier getClassifier() {
        return classifier;
    }

    public void setClassifier(AbstractClassifier classifier) {
        this.classifier = classifier;
    }

    public void setupTestTunedClassifier() {         
        //setup classifier. in this example, if we wanted to tune the kernal as well, 
        //we'd have to extend smo and override the setOptions to allow specific options
        //for kernal settings... see SMO.setOptions()
        SMO svm = new SMO();
        PolyKernel p=new PolyKernel();
        p.setExponent(2);
        svm.setKernel(p);
        this.classifier = new SMO();                
        
        //setup tuner, defaults to 10foldCV grid-search
        this.tuner = new Tuner(); 
        
        //setup para space 
        int size = 5;
        double[] cs = new double[size];
        for (int i = 0; i < cs.length; i++)
            cs[i] = Math.pow(10.0, (i-size/2)*2);
        
        this.space = new ParameterSpace();
        space.addParameter("C", cs);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (tuner == null)
            throw new Exception("Tuner not setup");
        if (space == null)
            throw new Exception("Parameter space not setup");
        if (classifier == null)
            throw new Exception("No classifier specified");
        
        ParameterSet bestParas = tuner.tune(classifier, data, space);
        
        String[] options = bestParas.toOptionsList();
        classifier.setOptions(options);
        classifier.buildClassifier(data);
    }
 
    @Override
    public double[] distributionForInstance(Instance inst) throws Exception { 
        return classifier.distributionForInstance(inst);
    }
    
    public static void main(String[] args) throws Exception {
        String dataset = "bank";
        
        Instances all = ClassifierTools.loadData("Z:\\Data\\UCIDelgado\\"+dataset+"\\"+dataset+".arff");
        Instances[] data = InstanceTools.resampleInstances(all, 0, 0.5);
        
        TunedClassifier tc = new TunedClassifier();
        tc.setupTestTunedClassifier();
        
        Classifier[] cs = new Classifier[] { new SMO(), tc };
        
        for (Classifier c : cs) {
            c.buildClassifier(data[0]);
            System.out.println(ClassifierTools.accuracy(data[1], c));
        }
        
    }
}
