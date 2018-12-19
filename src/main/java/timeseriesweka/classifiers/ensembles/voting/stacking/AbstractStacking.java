package timeseriesweka.classifiers.ensembles.voting.stacking;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.logging.Level;
import weka.classifiers.Classifier;
import utilities.ClassifierResults;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import timeseriesweka.classifiers.ensembles.voting.ModuleVotingScheme;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;


/**
 *
 * A given classifier is trained on a set of instances where each inst is formed by
 * concatenating the weighted distforinst output of each module for this instance. For 
 * unweighted distforinsts to be considered, can use of course just EqualWeighting()
 * 
 * @author James Large
 */
public abstract class AbstractStacking extends ModuleVotingScheme {
    
    protected Classifier classifier;
    protected int numOutputAtts;
    protected Instances instsHeader;
    
    public AbstractStacking(Classifier classifier) {
        this.classifier = classifier;
    }
    
    public AbstractStacking(Classifier classifier, int numClasses) {
        this.classifier = classifier;
        this.numClasses = numClasses;
    }
    
    public Classifier getClassifier() { 
        return classifier;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) throws Exception {
        this.numClasses = numClasses;
        setNumOutputAttributes(modules);
        int numInsts = modules[0].trainResults.numInstances();
        
        initInstances();
        Instances insts = new Instances(this.instsHeader, numInsts);
        
        for (int i = 0; i < numInsts; i++) 
            insts.add(buildInst(modules, true, i));

        classifier.buildClassifier(insts);
    }
    
    protected abstract void setNumOutputAttributes(EnsembleModule[] modules) throws Exception;
    protected abstract Instance buildInst(double[][] dists, Double classVal) throws Exception;
    
    protected Instance buildInst(EnsembleModule[] modules, boolean train, int instIndex)  throws Exception {
        double[][] dists = new double[modules.length][];
        
        for (int m = 0; m < modules.length; m++) {
            if (train)
                dists[m] = modules[m].trainResults.getDistributionForInstance(instIndex);
            else //test
                dists[m] = modules[m].testResults.getDistributionForInstance(instIndex);
            
            for (int c = 0; c < numClasses; c++) 
                dists[m][c] *= modules[m].priorWeight * modules[m].posteriorWeights[c];
        }
        
        Double classVal = train ? modules[0].trainResults.getTrueClassValue(instIndex) : null;
        return buildInst(dists, classVal);
    }

    protected void initInstances() {
        ArrayList<Attribute> atts = new ArrayList<>(numOutputAtts);
        for (int i = 0; i < numOutputAtts-1; i++)
            atts.add(new Attribute(""+i));
        
        ArrayList<String> classVals = new ArrayList<>(numClasses);
        for (int i = 0; i < numClasses; i++)
            classVals.add("" + i);
        atts.add(new Attribute("class", classVals));
        
        instsHeader = new Instances("", atts, 1);
        instsHeader.setClassIndex(numOutputAtts-1);
    }
    
    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) throws Exception {
        Instance inst = buildInst(modules, true, trainInstanceIndex);
        return classifier.distributionForInstance(inst);
    }
    
    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) throws Exception {
        Instance inst = buildInst(modules, false, testInstanceIndex);
        
        return classifier.distributionForInstance(inst);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[][] dists = new double[modules.length][];
        
        for(int m = 0; m < modules.length; m++){
            dists[m] = modules[m].getClassifier().distributionForInstance(testInstance); 
            storeModuleTestResult(modules[m], dists[m]);
            
            for (int c = 0; c < numClasses; c++) 
                dists[m][c] *= modules[m].priorWeight * modules[m].posteriorWeights[c];
        }
        
        Instance inst = buildInst(dists, null);
        return classifier.distributionForInstance(inst);
    }
    
    public String toString() { 
        return super.toString() + "(" + classifier.getClass().getSimpleName() + ")";
    }
    
}
