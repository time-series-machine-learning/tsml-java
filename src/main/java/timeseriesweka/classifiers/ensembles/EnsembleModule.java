package timeseriesweka.classifiers.ensembles;

import utilities.ClassifierResults;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import utilities.DebugPrinting;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * A little class to store information about a (unspecified) classifier's results on a (unspecified) dataset
 * Used in the ensemble classes HESCA and EnsembleFromFile to store loaded results
 * 
 * Will be expanded in the future
 * 
 * @author James Large
 */
public class EnsembleModule implements DebugPrinting {
    
    
    private Classifier classifier;

    private String moduleName;
    private String parameters;
    
    public ClassifierResults trainResults;
    public ClassifierResults testResults;
    
    private int numClasses;
    
    //by default (and i imagine in the vast majority of cases) all prior weights are equal (i.e 1)
    //however may be circumstances where certain classifiers are themselves part of 
    //a subensemble or something
    public double priorWeight = 1.0; 
    
    //each module makes a vote, with a weight defined for this classifier when predicting this class 
    //many weighting schemes will have weights for each class set to a single classifier equal, but some 
    //will have e.g certain members being experts at classifying certain classes etc
    public double[] posteriorWeights;

    public EnsembleModule() {
        this.moduleName = "ensembleModule";
        this.classifier = null;
        
        trainResults = null;
        testResults = null;
    }
    
    public EnsembleModule(String moduleName, Classifier classifier, String parameters) {
        this.classifier =classifier;
        this.moduleName = moduleName;
        this.parameters = parameters;
        
        trainResults = null;
        testResults = null;
    }

    public String getModuleName() {
        return moduleName;
    }

    public void setModuleName(String moduleName) {
        this.moduleName = moduleName;
    }

    public String getParameters() {
        return parameters;
    }

    public void setParameters(String parameters) {
        this.parameters = parameters;
    }
    
    public Classifier getClassifier() {
        return classifier;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }
    
    @Override
    public String toString() {
        return moduleName;
    }
}
