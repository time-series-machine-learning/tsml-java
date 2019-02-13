
package vector_classifiers;

import evaluation.ClassifierResults;
import evaluation.tuning.ParameterResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import evaluation.tuning.Tuner;
import evaluation.tuning.evaluators.StratifiedResamplesEvaluator;
import evaluation.tuning.searchers.RandomSearcher;
import timeseriesweka.classifiers.ParameterSplittable;
import timeseriesweka.classifiers.SaveParameterInfo;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Given 
 *      1) a tuning method, 
 *              - DEFAULT: tuner set to gridsearch + 10foldcv 
 *              - settable via setTuner(...)
 *      2) a base classifier with a well formed setOptions(String[]) method 
 *          (that must extend AbstractClassifier, the Classifier interface 
 *          alone does not provide setOptions()) 
 *              - DEFAULT: none
 *              - settable via setClassifier(...)
 *      3) a parameter space expressed as keys (that correspond to flags in the setOptions() method) 
 *          to lists of values that these parameters can take.
 *              - DEFAULT: none
 *              - settable via setParameterSpace(...)
 * 
 * For a basic example of the above, see setupTestTunedClassifier()
 * 
 * This class will select the best parameter set on the given dataset according to the 
 * selection and evaluation methods described by the tuner, and build the base classifier
 * with the best parameters found
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class TunedClassifier extends AbstractClassifier implements SaveParameterInfo, TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable {

    int seed;
    ParameterSpace space = null;
    Tuner tuner = null;
    AbstractClassifier classifier = null;

    ParameterSet bestParas = null;

    //interface variables
    ClassifierResults trainResults = null;
    double trainAcc = 0;
    
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
    
    void setSeed(int seed) { 
        this.seed = seed;
        
        tuner.setSeed(seed);
        //no setSeed in abstractclassifier. i imagine most define it via setOptions,
        //so could add it a a parameter with only one possible value, or jsut set the seed
        //before giving the classifier to this tunedclassifier instance
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

    public void setParameterSpace(ParameterSpace space) {
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
        int size = 9;
        double[] cs = new double[size];
        for (int i = 0; i < cs.length; i++)
            cs[i] = Math.pow(10.0, (i-size/2));
        
        this.space = new ParameterSpace();
        this.space.addParameter("C", cs);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        //check everything's here
        boolean somethingMissing = false;
        String msg = "";
        if (tuner == null) {
            msg += "Tuner not setup. ";
            somethingMissing = true;
        }
        if (space == null) {
            msg += "Parameter space not setup. ";
            somethingMissing = true;
        }
        if (classifier == null) {
            msg += "No classifier specified. ";
            somethingMissing = true;
        }
        if (somethingMissing) 
            throw new Exception("TunedClassifier: " + msg);
        
        //actual work
        ParameterResults bestParas = tuner.tune(classifier, data, space);
        
        trainResults = bestParas.results;
        trainAcc = bestParas.results.acc;
        
        String[] options = bestParas.paras.toOptionsList();
        classifier.setOptions(options);
        classifier.buildClassifier(data);
    }
 
    @Override
    public double[] distributionForInstance(Instance inst) throws Exception { 
        return classifier.distributionForInstance(inst);
    }
    
    public static void main(String[] args) throws Exception {
        String dataset = "hayes-roth";
        
        TunedClassifier tcGrid = new TunedClassifier();
        tcGrid.setupTestTunedClassifier();
        tcGrid.setCloneClassifierForEachParameterEval(false);
        
        TunedClassifier tcRand = new TunedClassifier();
        tcRand.setupTestTunedClassifier();
        tcRand.getTuner().setSearcher(new RandomSearcher(3));
        tcRand.getTuner().setEvaluator(new StratifiedResamplesEvaluator());
        tcRand.setCloneClassifierForEachParameterEval(false);
        
        
        Classifier[] cs = new Classifier[] { tcRand, new SMO(), tcGrid };
        
        int numFolds = 10;
        
        for (Classifier c : cs) {
            Instances all = ClassifierTools.loadData("Z:\\Data\\UCIDelgado\\"+dataset+"\\"+dataset+".arff");
            double mean =.0;
            
            for (int f = 0; f < numFolds; f++) {
                Instances[] data = InstanceTools.resampleInstances(all, f, 0.5);
                
                try {
                    ((TunedClassifier)c).setSeed(f);
                }catch (Exception e){ }
                     
                c.buildClassifier(data[0]);
                double t = ClassifierTools.accuracy(data[1], c);
                mean += t;
                System.out.print(t + ", ");
            }
            
            mean /= numFolds;
            System.out.println("\nmean = " + mean);
        }
        
    }

    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    // METHODS FOR:               SaveParameterInfo, TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable
    
    @Override //SaveParameterInfo
    public String getParameters() {
        return bestParas.toClassifierResultsParaLine(true);
    }

    @Override //TrainAccuracyEstimate
    public void setFindTrainAccuracyEstimate(boolean setCV) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override //TrainAccuracyEstimate
    public void writeCVTrainToFile(String train) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override //TrainAccuracyEstimate
    public ClassifierResults getTrainResults() {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override //SaveEachParameter
    public void setPathToSaveParameters(String r) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override //SaveEachParameter
    public void setSaveEachParaAcc(boolean b) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override //ParameterSplittable
    public void setParamSearch(boolean b) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override //ParameterSplittable
    public void setParametersFromIndex(int x) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override //ParameterSplittable
    public String getParas() {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

    @Override //ParameterSplittable
    public double getAcc() {
        throw new UnsupportedOperationException("Not supported yet."); 
    }
}
