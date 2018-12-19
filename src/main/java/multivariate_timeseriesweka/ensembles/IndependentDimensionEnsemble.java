/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka.ensembles;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import timeseriesweka.classifiers.ensembles.voting.MajorityVote;
import timeseriesweka.classifiers.ensembles.voting.ModuleVotingScheme;
import timeseriesweka.classifiers.ensembles.weightings.EqualWeighting;
import timeseriesweka.classifiers.ensembles.weightings.ModuleWeightingScheme;
import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstanceWithClassVal;
import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class IndependentDimensionEnsemble extends AbstractClassifier{
    
    protected ModuleWeightingScheme weightingScheme = new EqualWeighting();
    protected ModuleVotingScheme votingScheme = new MajorityVote();
    protected EnsembleModule[] modules;
    
    long seed;
    
    int numClasses, numChannels;
    Instances train;
    Instances[] channels;
    Classifier[] classifiers;
    String[] classifierNames;
    
    Classifier original_model;
    
    double[] priorWeights;
    
    public IndependentDimensionEnsemble(Classifier cla){
        original_model = cla;
    }
   
    public void setSeed(long sd){
        seed = sd;
        
        if(original_model instanceof RandomizableIteratedSingleClassifierEnhancer){
            RandomizableIteratedSingleClassifierEnhancer r = (RandomizableIteratedSingleClassifierEnhancer) original_model;
            r.setSeed((int) seed);
        }
        else{ 
            //check through reflection if the classifier has a method with seed in the name, that takes an int or a long.
            Method[] methods = original_model.getClass().getMethods();
            for (Method method : methods) {
                Class[] paras = method.getParameterTypes();
                //if the method contains the name seed, and takes in 1 parameter, thats a primitive. probably setRandomSeed.
                String name = method.getName().toLowerCase();
                if((name.contains("random") || name.contains("seed"))
                    && paras.length == 1 && (paras[0] == int.class || paras[0] == long.class )){
                    try {
                        if(paras[0] == int.class)
                            method.invoke(original_model, (int) seed);
                        else
                            method.invoke(original_model, seed);
                        
                    } catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
                        System.out.println(ex);
                        System.out.println("Tried to set the seed method name: " + method.getName());
                    }
                }
            }
        }
        
    }
    
    public void setPriorWeights(double[] weights){
        priorWeights = weights;
    }
    
    protected void initialiseModules() throws Exception{
        classifiers = AbstractClassifier.makeCopies(original_model, numChannels);
        classifierNames = new String[numChannels];
        
        //one module for each channel.
        this.modules = new EnsembleModule[numChannels];
        for (int m = 0; m < numChannels; m++){
            classifierNames[m] = classifiers[m].getClass().getSimpleName() +"_"+m;
            modules[m] = new EnsembleModule(classifierNames[m], classifiers[m], "");
            if(priorWeights != null)
                modules[m].priorWeight = priorWeights[m];
        }
        
        weightingScheme.defineWeightings(modules, numClasses);
        votingScheme.trainVotingScheme(modules, numClasses);
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        train = data;
        numClasses = data.numClasses();
        channels = splitMultivariateInstances(data);
        numChannels = channels.length;
        initialiseModules();
               
        //build the classifier.
        for(int i=0; i<numChannels; i++){
            Instances channel = channels[i];
            modules[i].getClassifier().buildClassifier(channel);
        }
    }
    
    Instances[] convertedTest = null;
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {   
        double[] dist = distributionForInstance(votingScheme, modules, splitMultivariateInstanceWithClassVal(instance));
        return dist;
    }

    /*@Override
    public String toString(){
        String output = "";
        for(Classifier cl : classifiers)
            output += cl.toString() + ",";
        return output;
    }*/
    
    private double[] distributionForInstance(ModuleVotingScheme vs, EnsembleModule[] modules, Instance[] testInstance) throws Exception{
        double[] preds = new double[numClasses];
        
        int pred;
        double[] dist;
        for(int m = 0; m < numChannels; m++){
            dist = modules[m].getClassifier().distributionForInstance(testInstance[m]); 
            vs.storeModuleTestResult(modules[m], dist);
            
            pred = (int)vs.indexOfMax(dist);
            preds[pred] += modules[m].priorWeight * 
                           modules[m].posteriorWeights[pred];
        }
        
        return vs.normalise(preds);
    }
}
