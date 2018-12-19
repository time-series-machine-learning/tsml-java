/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka.classifiers;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;
import static utilities.multivariate_tools.MultivariateInstanceTools.concatinateInstances;
import static utilities.multivariate_tools.MultivariateInstanceTools.numChannels;
import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstanceWithClassVal;
import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstances;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.core.Instance;

/**
 *
 * @author raj09hxu
 */
public class ConcatenateClassifier extends AbstractClassifier{

    Classifier original_model;
    Instances concat_train;
    Instances concat_test;
    
    long seed;
    
    public ConcatenateClassifier(Classifier cla){
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
    
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        concat_train = concatinateInstances(splitMultivariateInstances(data));
        original_model.buildClassifier(concat_train);
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {   
        
        //split the multivariate test into 
        if(concat_test == null){
            concat_test = concatinateInstances(splitMultivariateInstances(instance.dataset()));
        }
        
        //get the index of the text instance from the original and use that in the concatenated.
        double[] dist = original_model.distributionForInstance(concat_test.get(instance.dataset().indexOf(instance)));
        return dist;
    }

    
    
}
