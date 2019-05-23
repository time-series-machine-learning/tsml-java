/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import static experiments.ClassifierLists.setClassifierClassic;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransform;
import weka.classifiers.Classifier;
import weka.filters.SimpleBatchFilter;

/**
 *
 * @author a.bostrom1
 */
public class TransformLists {
 
    
    public static SimpleBatchFilter setTransform(Experiments.ExperimentalArguments exp){
        return setClassicTransform(exp.classifierName, exp.foldId);
    }

    public static SimpleBatchFilter setClassicTransform(String classifierName, int foldId) {
        SimpleBatchFilter transformer = null;
        switch(classifierName){
            case "ShapeletTransform": case "ST":
                transformer = new ShapeletTransform();
                break;
            default:
                System.out.println("UNKNOWN CLASSIFIER "+classifierName);
                System.exit(0);
        }
        
        return transformer;
    }
    
   public static void main(String[] args) throws Exception {
        System.out.println(setClassicTransform("ST", 0));
        System.out.println(setClassicTransform("ShapeletTransform", 0));
    }
    
    
    
}
